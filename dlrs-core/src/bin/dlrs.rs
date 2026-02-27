//! DLRS CLI — LoRA Multi-Function Manager with Auto ZK Sharing
//!
//! Commands:
//!   dlrs create   — create a new LoRA adapter
//!   dlrs list     — list all managed adapters
//!   dlrs evolve   — run evolution on an adapter
//!   dlrs merge    — merge two adapters
//!   dlrs export   — export an adapter
//!   dlrs stats    — show manager statistics
//!   dlrs serve    — start P2P node with auto ZK sharing
//!   dlrs demo     — run a full demo (create, evolve, merge, share)

use dlrs_core::lora::{
    AdapterFormat, AutoShareDaemon, DaemonConfig, LoraConfig, LoraManager,
};
use dlrs_core::network::{run_swarm, SwarmConfig};
use nalgebra::DMatrix;
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;

const STORE_FILE: &str = "dlrs-store.json";

fn print_usage() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║        DLRS — Distributed Low-Rank Space                    ║
║        LoRA Multi-Function Manager + Auto ZK Sharing        ║
╚══════════════════════════════════════════════════════════════╝

Usage: dlrs <command> [options]

Commands:
  create  <name> <module> <rank> <m> <n> [domains...]  Create a new LoRA adapter
  list                                                  List all adapters
  evolve  <id> [steps]                                  Evolve an adapter
  merge   <id_a> <id_b>                                 Merge two adapters
  export  <id> [format: json|delta]                     Export an adapter
  stats                                                 Show manager statistics
  serve   [port]                                        Start P2P node + auto ZK sharing
  demo                                                  Run full interactive demo

Examples:
  dlrs create attention-q attn.q 8 768 768 nlp reasoning
  dlrs serve 9000
  dlrs demo
"#
    );
}

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "create" => cmd_create(&args[2..]).await,
        "list" => cmd_list().await,
        "evolve" => cmd_evolve(&args[2..]).await,
        "merge" => cmd_merge(&args[2..]).await,
        "export" => cmd_export(&args[2..]).await,
        "stats" => cmd_stats().await,
        "serve" => cmd_serve(&args[2..]).await,
        "demo" => cmd_demo().await,
        "help" | "--help" | "-h" => print_usage(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
        }
    }
}

/// Load or create manager
fn load_manager() -> LoraManager {
    match LoraManager::load(STORE_FILE) {
        Ok(mgr) => {
            println!("  Loaded {} adapters from {}", mgr.count(), STORE_FILE);
            mgr
        }
        Err(_) => {
            println!("  No existing store found, starting fresh");
            LoraManager::with_store(STORE_FILE)
        }
    }
}

/// Save manager
fn save_manager(mgr: &LoraManager) {
    if let Err(e) = mgr.save() {
        eprintln!("  Failed to save: {}", e);
    } else {
        println!("  Saved to {}", STORE_FILE);
    }
}

async fn cmd_create(args: &[String]) {
    if args.len() < 5 {
        eprintln!("Usage: dlrs create <name> <module> <rank> <m> <n> [domains...]");
        return;
    }

    let name = &args[0];
    let module = &args[1];
    let rank: usize = args[2].parse().expect("rank must be a number");
    let m: usize = args[3].parse().expect("m must be a number");
    let n: usize = args[4].parse().expect("n must be a number");
    let domains: Vec<String> = if args.len() > 5 {
        args[5..].to_vec()
    } else {
        vec!["general".to_string()]
    };

    let mut mgr = load_manager();
    let config = LoraConfig {
        name: name.clone(),
        target_module: module.clone(),
        rank,
        alpha: rank as f64 * 2.0,
        domains,
        original_dims: (m, n),
    };
    let id = mgr.create_adapter(config);
    println!("\n  Created adapter: {}", id);
    println!("  {}", mgr.get(&id).unwrap().summary());
    save_manager(&mgr);
}

async fn cmd_list() {
    let mgr = load_manager();
    if mgr.count() == 0 {
        println!("\n  No adapters. Use 'dlrs create' or 'dlrs demo' to get started.");
        return;
    }
    println!("\n  Adapters ({}):", mgr.count());
    println!("  {}", "-".repeat(80));
    for (id, _name) in mgr.list() {
        if let Some(adapter) = mgr.get(&id) {
            println!("  [{}] {}", &id[..8], adapter.summary());
        }
    }
}

async fn cmd_evolve(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: dlrs evolve <id-prefix> [steps]");
        return;
    }

    let prefix = &args[0];
    let steps: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);

    let mut mgr = load_manager();
    let matching: Vec<String> = mgr
        .list()
        .into_iter()
        .filter(|(id, _)| id.starts_with(prefix))
        .map(|(id, _)| id)
        .collect();

    if matching.is_empty() {
        eprintln!("  No adapter matching '{}'", prefix);
        return;
    }

    for id in matching {
        let adapter = mgr.get(&id).unwrap();
        let (m, n) = adapter.config.original_dims;
        mgr.evolution_config.steps_per_cycle = steps;
        let target = DMatrix::new_random(m, n) * 0.01;
        let fitness = mgr.evolve_adapter(&id, &target).unwrap();
        println!("  Evolved [{}]: fitness={:.4}", &id[..8], fitness);
    }
    save_manager(&mgr);
}

async fn cmd_merge(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: dlrs merge <id-prefix-a> <id-prefix-b>");
        return;
    }

    let mut mgr = load_manager();
    let find_id = |prefix: &str| -> Option<String> {
        mgr.list()
            .into_iter()
            .find(|(id, _)| id.starts_with(prefix))
            .map(|(id, _)| id)
    };

    let id_a = match find_id(&args[0]) {
        Some(id) => id,
        None => {
            eprintln!("  No adapter matching '{}'", args[0]);
            return;
        }
    };
    let id_b = match find_id(&args[1]) {
        Some(id) => id,
        None => {
            eprintln!("  No adapter matching '{}'", args[1]);
            return;
        }
    };

    if let Some(merged_id) = mgr.merge_adapters(&id_a, &id_b) {
        println!(
            "\n  Merged: {}",
            mgr.get(&merged_id).unwrap().summary()
        );
        save_manager(&mgr);
    }
}

async fn cmd_export(args: &[String]) {
    if args.is_empty() {
        eprintln!("Usage: dlrs export <id-prefix> [json|delta]");
        return;
    }

    let mgr = load_manager();
    let prefix = &args[0];
    let format = match args.get(1).map(|s| s.as_str()) {
        Some("delta") => AdapterFormat::WeightDelta,
        _ => AdapterFormat::DlrsJson,
    };

    let matching: Vec<String> = mgr
        .list()
        .into_iter()
        .filter(|(id, _)| id.starts_with(prefix))
        .map(|(id, _)| id)
        .collect();

    for id in matching {
        let adapter = mgr.get(&id).unwrap();
        match adapter.export(&format) {
            Ok(json) => {
                let filename = format!("dlrs-export-{}.json", &id[..8]);
                std::fs::write(&filename, &json).expect("Failed to write export");
                println!("  Exported [{}] -> {}", &id[..8], filename);
            }
            Err(e) => eprintln!("  Export error: {}", e),
        }
    }
}

async fn cmd_stats() {
    let mgr = load_manager();
    let stats = mgr.stats();
    println!("\n  DLRS Manager Statistics");
    println!("  {}", "=".repeat(40));
    println!("  Adapters:     {}", stats.total_adapters);
    println!("  Avg Fitness:  {:.4}", stats.avg_fitness);
    println!("  Best Fitness: {:.4}", stats.best_fitness);
    println!("  Domains:      {}", stats.total_domains);
    println!("  Parameters:   {}", stats.total_parameters);
}

async fn cmd_serve(args: &[String]) {
    let port: u16 = args.first().and_then(|s| s.parse().ok()).unwrap_or(0);

    println!("\n  Starting DLRS P2P Node...");
    println!("  {}", "=".repeat(50));

    let mgr = load_manager();
    let manager = Arc::new(Mutex::new(mgr));

    // Start the libp2p swarm
    let swarm_config = SwarmConfig {
        listen_port: port,
        heartbeat_secs: 1,
    };

    let (peer_id, cmd_tx, evt_rx) = match run_swarm(swarm_config).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("  Failed to start P2P swarm: {}", e);
            return;
        }
    };

    println!("  Peer ID: {}", peer_id);
    println!("  Listening for peers via mDNS...");

    // Start the auto-sharing daemon
    let daemon_config = DaemonConfig {
        announce_interval_secs: 15,
        evolution_interval_secs: 60,
        lifecycle_interval_secs: 120,
        peer_id: peer_id.to_string(),
    };

    let daemon = AutoShareDaemon::new(daemon_config, manager.clone(), cmd_tx.clone());
    daemon.run(evt_rx).await;

    println!("  Auto ZK sharing active. Press Ctrl+C to stop.\n");

    // Keep running until interrupted
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for Ctrl+C");

    println!("\n  Shutting down...");
    let _ = cmd_tx.send(dlrs_core::network::SwarmCommand::Shutdown).await;

    // Save state
    let mgr = manager.lock().await;
    save_manager(&mgr);
    println!("  Goodbye!");
}

async fn cmd_demo() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║              DLRS — Full Demo                               ║
║       LoRA Management + Evolution + ZK Sharing              ║
╚══════════════════════════════════════════════════════════════╝
"#
    );

    // Step 1: Create adapters
    println!("Step 1: Creating LoRA adapters...");
    println!("{}", "-".repeat(60));
    let mut mgr = LoraManager::with_store(STORE_FILE);

    let attn_q_id = mgr.create_adapter(LoraConfig {
        name: "attn-query".to_string(),
        target_module: "transformer.attention.query".to_string(),
        rank: 8,
        alpha: 16.0,
        domains: vec!["nlp".into(), "reasoning".into()],
        original_dims: (256, 256),
    });
    println!("  {}", mgr.get(&attn_q_id).unwrap().summary());

    let attn_v_id = mgr.create_adapter(LoraConfig {
        name: "attn-value".to_string(),
        target_module: "transformer.attention.value".to_string(),
        rank: 8,
        alpha: 16.0,
        domains: vec!["nlp".into(), "memory".into()],
        original_dims: (256, 256),
    });
    println!("  {}", mgr.get(&attn_v_id).unwrap().summary());

    let ffn_id = mgr.create_adapter(LoraConfig {
        name: "ffn-up".to_string(),
        target_module: "transformer.ffn.up".to_string(),
        rank: 16,
        alpha: 32.0,
        domains: vec!["vision".into(), "multimodal".into()],
        original_dims: (256, 1024),
    });
    println!("  {}", mgr.get(&ffn_id).unwrap().summary());

    // Step 2: Evolution
    println!("\nStep 2: Evolving adapters (simulating fine-tuning)...");
    println!("{}", "-".repeat(60));

    for (id, name) in [(&attn_q_id, "attn-query"), (&attn_v_id, "attn-value"), (&ffn_id, "ffn-up")] {
        let adapter = mgr.get(id).unwrap();
        let (m, n) = adapter.config.original_dims;
        let target = DMatrix::new_random(m, n) * 0.01;
        mgr.evolution_config.steps_per_cycle = 20;
        let fitness = mgr.evolve_adapter(id, &target).unwrap();
        println!("  {} evolved -> fitness={:.4}, epoch={}", name, fitness, mgr.get(id).unwrap().seed.epoch);
    }

    // Step 3: ZK Capability Proofs
    println!("\nStep 3: Generating ZK capability proofs...");
    println!("{}", "-".repeat(60));

    for (id, name) in [(&attn_q_id, "attn-query"), (&attn_v_id, "attn-value"), (&ffn_id, "ffn-up")] {
        let n = mgr.get(id).unwrap().seed.lrim.n;
        let domain_vec = nalgebra::DVector::new_random(n);
        let domain_name = &mgr.get(id).unwrap().config.domains[0];
        if let Some(proof) = mgr.generate_proof(id, domain_name, &domain_vec) {
            println!(
                "  {} | domain='{}' | score={:.4} | proof={}...",
                name,
                domain_name,
                proof.claimed_score,
                &proof.proof_hash[..16]
            );
        }
    }

    // Step 4: Merge adapters
    println!("\nStep 4: Merging attention adapters...");
    println!("{}", "-".repeat(60));

    let merged_id = mgr.merge_adapters(&attn_q_id, &attn_v_id).unwrap();
    println!("  {}", mgr.get(&merged_id).unwrap().summary());

    // Step 5: Apply to base weights
    println!("\nStep 5: Applying stacked LoRA to base weights...");
    println!("{}", "-".repeat(60));

    let base_weights = DMatrix::new_random(256, 256);
    let base_norm = base_weights.norm();
    let modified = mgr.apply_stack(&[&attn_q_id, &attn_v_id], &base_weights);
    let delta_norm = (&modified - &base_weights).norm();
    println!(
        "  Base norm: {:.2} | Delta norm: {:.4} | Ratio: {:.6}",
        base_norm,
        delta_norm,
        delta_norm / base_norm
    );

    // Step 6: Stats
    println!("\nStep 6: Manager statistics...");
    println!("{}", "-".repeat(60));
    let stats = mgr.stats();
    println!("  Adapters:     {}", stats.total_adapters);
    println!("  Avg Fitness:  {:.4}", stats.avg_fitness);
    println!("  Best Fitness: {:.4}", stats.best_fitness);
    println!("  Domains:      {}", stats.total_domains);
    println!("  Parameters:   {}", stats.total_parameters);

    // Step 7: Persistence
    println!("\nStep 7: Saving state...");
    println!("{}", "-".repeat(60));
    save_manager(&mgr);

    // Step 8: P2P sharing
    println!("\nStep 8: Starting P2P node for ZK sharing...");
    println!("{}", "-".repeat(60));

    let manager = Arc::new(Mutex::new(mgr));

    let swarm_config = SwarmConfig {
        listen_port: 0,
        heartbeat_secs: 1,
    };

    match run_swarm(swarm_config).await {
        Ok((peer_id, cmd_tx, evt_rx)) => {
            println!("  Peer ID: {}", peer_id);
            println!("  P2P node running, announcing adapters...");

            let daemon = AutoShareDaemon::new(
                DaemonConfig {
                    announce_interval_secs: 5,
                    evolution_interval_secs: 30,
                    lifecycle_interval_secs: 60,
                    peer_id: peer_id.to_string(),
                },
                manager.clone(),
                cmd_tx.clone(),
            );
            daemon.run(evt_rx).await;

            // Let it run for a few seconds to demonstrate
            println!("  Listening for peers for 10 seconds...");
            tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;

            let _ = cmd_tx
                .send(dlrs_core::network::SwarmCommand::Shutdown)
                .await;
            println!("  P2P node stopped.");
        }
        Err(e) => {
            println!("  P2P not available in this environment: {}", e);
            println!("  (This is expected in sandboxed environments)");
        }
    }

    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║              Demo Complete!                                  ║
║                                                              ║
║  Created 3 LoRA adapters, evolved them, generated ZK proofs, ║
║  merged attention adapters, applied stacked LoRA, and ran    ║
║  P2P auto-sharing with gossipsub + mDNS discovery.           ║
║                                                              ║
║  Run 'dlrs serve' to start a persistent P2P node.            ║
╚══════════════════════════════════════════════════════════════╝
"#
    );
}
