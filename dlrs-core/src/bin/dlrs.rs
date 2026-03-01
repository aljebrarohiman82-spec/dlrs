//! DLRS CLI — LoRA Multi-Function Manager with Auto ZK Sharing + TEE Security
//!
//! Commands:
//!   dlrs create     — create a new LoRA adapter
//!   dlrs list       — list all managed adapters
//!   dlrs evolve     — run evolution on an adapter
//!   dlrs merge      — merge two adapters
//!   dlrs export     — export an adapter
//!   dlrs stats      — show manager statistics
//!   dlrs train      — train a LoRA adapter on a dataset
//!   dlrs auto-train — AI-assisted automatic training
//!   dlrs backup     — create/list/restore backup snapshots
//!   dlrs tee        — TEE enclave management and secure operations
//!   dlrs serve      — start P2P node with auto ZK sharing
//!   dlrs demo       — run a full demo (train, evolve, backup, share, TEE)

use dlrs_core::lora::{
    AdapterFormat, AutoShareDaemon, DaemonConfig, LoraConfig, LoraManager,
    Dataset, TrainConfig, LossFunction, LrSchedule, train, auto_train,
    SharingRegistry,
};
use dlrs_core::storage::backup::BackupManager;
use dlrs_core::tee::{
    TeeEnclave, TeeBackend, SealedStorage,
    AttestationPolicy, SecureChannel,
    generate_quote, verify_quote,
};
use dlrs_core::network::{run_swarm, SwarmConfig};
use nalgebra::DMatrix;
use std::env;
use std::sync::Arc;
use tokio::sync::Mutex;

const STORE_FILE: &str = "dlrs-store.json";
const BACKUP_DIR: &str = "dlrs-backups";
const SEALED_DIR: &str = "dlrs-sealed";

fn print_usage() {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║        DLRS v1.0 — Distributed Low-Rank Space               ║
║        LoRA Multi-Function Manager + Auto ZK Sharing         ║
╚══════════════════════════════════════════════════════════════╝

Usage: dlrs <command> [options]

Commands:
  create     <name> <module> <rank> <m> <n> [domains...]  Create a new LoRA adapter
  list                                                     List all adapters
  evolve     <id> [steps]                                  Evolve an adapter
  merge      <id_a> <id_b>                                 Merge two adapters
  export     <id> [format: json|delta]                     Export an adapter
  stats                                                    Show manager statistics
  train      <id> <dataset> [epochs] [lr]                  Train adapter on dataset
  auto-train <name> <module> <dataset> [domains...]        AI-assisted automatic training
  backup     [create|list|restore <ver>|verify]            Backup management
  tee        [init|status|seal <id>|attest|channel]        TEE enclave operations
  serve      [port]                                        Start P2P node + auto ZK sharing
  demo                                                     Run full interactive demo

Examples:
  dlrs create attention-q attn.q 8 768 768 nlp reasoning
  dlrs train <id> my-data.json 100 0.01
  dlrs auto-train smart-lora attn.q dataset.json nlp reasoning
  dlrs backup create
  dlrs backup restore 3
  dlrs tee init sgx
  dlrs tee seal <adapter-id>
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
        "train" => cmd_train(&args[2..]).await,
        "auto-train" => cmd_auto_train(&args[2..]).await,
        "backup" => cmd_backup(&args[2..]).await,
        "tee" => cmd_tee(&args[2..]).await,
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

async fn cmd_train(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: dlrs train <id-prefix> <dataset-file> [epochs] [lr]");
        return;
    }

    let prefix = &args[0];
    let dataset_path = &args[1];
    let epochs: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let lr: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.01);

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

    let dataset = match Dataset::load(dataset_path) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("  Failed to load dataset '{}': {}", dataset_path, e);
            eprintln!("  Generating synthetic dataset for demonstration...");
            let adapter = mgr.get(&matching[0]).unwrap();
            let (m, n) = adapter.config.original_dims;
            Dataset::synthetic("synthetic", n, m, 50)
        }
    };

    let config = TrainConfig {
        epochs,
        learning_rate: lr,
        loss_fn: LossFunction::MSE,
        lr_schedule: LrSchedule::CosineAnnealing { min_lr: lr * 0.01 },
        patience: 30,
        min_delta: 1e-7,
        log_interval: (epochs / 10).max(1),
    };

    for id in matching {
        let adapter = mgr.get_mut(&id).unwrap();
        println!("\n  Training [{}] '{}'...", &id[..8], adapter.config.name);
        let history = train(adapter, &dataset, &config);
        println!(
            "  Done: best_loss={:.6} at epoch {} | final_loss={:.6} | early_stop={}",
            history.best_loss, history.best_epoch, history.final_loss, history.stopped_early
        );
    }
    save_manager(&mgr);
}

async fn cmd_auto_train(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: dlrs auto-train <name> <module> <dataset-file> [domains...]");
        return;
    }

    let name = &args[0];
    let module = &args[1];
    let dataset_path = &args[2];
    let domains: Vec<String> = if args.len() > 3 {
        args[3..].to_vec()
    } else {
        vec!["general".into()]
    };

    let dataset = match Dataset::load(dataset_path) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("  Failed to load dataset '{}': {}", dataset_path, e);
            eprintln!("  Generating synthetic dataset for demonstration...");
            Dataset::synthetic("synthetic", 64, 64, 50)
        }
    };

    println!("\n  Auto-training '{}' on '{}'...", name, dataset.name);
    let (adapter, history, rec) = auto_train(name, module, &dataset, domains);

    println!("\n  Recommendation: {}", rec.explanation);
    println!(
        "  Result: rank={} | lr={:.5} | best_loss={:.6} | epochs_run={} | early_stop={}",
        rec.rank,
        rec.learning_rate,
        history.best_loss,
        history.records.len(),
        history.stopped_early,
    );
    println!("  {}", adapter.summary());

    // Add to manager
    let mut mgr = load_manager();
    let id = adapter.id().to_string();
    mgr.add_adapter(adapter);
    println!("  Added to manager: {}", &id[..8]);
    save_manager(&mgr);
}

async fn cmd_backup(args: &[String]) {
    let subcmd = args.first().map(|s| s.as_str()).unwrap_or("create");

    match subcmd {
        "create" => {
            let mgr = load_manager();
            let data = match mgr.to_json() {
                Ok(json) => json,
                Err(e) => {
                    eprintln!("  Failed to serialize: {}", e);
                    return;
                }
            };
            let mut backup = BackupManager::new(BACKUP_DIR);
            let desc = args.get(1).map(|s| s.as_str()).unwrap_or("manual backup");
            match backup.create_snapshot(&data, mgr.count(), desc) {
                Ok(meta) => {
                    println!(
                        "\n  Backup created: v{} ({} bytes, {} adapters)",
                        meta.version, meta.size_bytes, meta.adapter_count
                    );
                    println!("  Checksum: {}", &meta.checksum[..16]);
                }
                Err(e) => eprintln!("  Backup failed: {}", e),
            }
        }
        "list" => {
            let backup = BackupManager::new(BACKUP_DIR);
            let snapshots = backup.list_snapshots();
            if snapshots.is_empty() {
                println!("\n  No backups found. Run 'dlrs backup create' first.");
                return;
            }
            println!("\n  Backups ({}):", snapshots.len());
            println!("  {}", "-".repeat(70));
            for snap in snapshots {
                println!(
                    "  v{:>4} | {} | {} adapters | {} bytes | {}",
                    snap.version,
                    snap.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    snap.adapter_count,
                    snap.size_bytes,
                    snap.description
                );
            }
            println!("  Total size: {} bytes", backup.total_size());
        }
        "restore" => {
            let version: u64 = match args.get(1).and_then(|s| s.parse().ok()) {
                Some(v) => v,
                None => {
                    eprintln!("Usage: dlrs backup restore <version>");
                    return;
                }
            };
            let backup = BackupManager::new(BACKUP_DIR);
            match backup.load_snapshot(version) {
                Ok(data) => {
                    if let Err(e) = std::fs::write(STORE_FILE, &data) {
                        eprintln!("  Restore failed: {}", e);
                    } else {
                        println!("  Restored v{} -> {}", version, STORE_FILE);
                    }
                }
                Err(e) => eprintln!("  Restore failed: {}", e),
            }
        }
        "verify" => {
            let backup = BackupManager::new(BACKUP_DIR);
            let results = backup.verify_all();
            if results.is_empty() {
                println!("\n  No backups to verify.");
                return;
            }
            println!("\n  Verification results:");
            for (ver, ok) in &results {
                println!("  v{}: {}", ver, if *ok { "OK" } else { "CORRUPTED" });
            }
            let all_ok = results.iter().all(|(_, ok)| *ok);
            println!(
                "  Overall: {}",
                if all_ok { "All backups intact" } else { "Some backups corrupted!" }
            );
        }
        other => {
            eprintln!("Unknown backup subcommand: {}", other);
            eprintln!("Usage: dlrs backup [create|list|restore <ver>|verify]");
        }
    }
}

async fn cmd_tee(args: &[String]) {
    let subcmd = args.first().map(|s| s.as_str()).unwrap_or("status");

    match subcmd {
        "init" => {
            let backend = match args.get(1).map(|s| s.as_str()) {
                Some("sgx") => TeeBackend::IntelSgx,
                Some("trustzone") => TeeBackend::ArmTrustZone,
                _ => TeeBackend::Simulated,
            };

            println!("\n  Initializing TEE enclave ({})...", backend.name());
            match TeeEnclave::new(backend) {
                Ok(enclave) => {
                    let status = enclave.status();
                    println!("  Enclave ID:      {}", &status.enclave_id[..8]);
                    println!("  Backend:         {}", backend.name());
                    println!("  Security Level:  {:?}", status.security_level);
                    println!("  Hardware TEE:    {}", backend.is_hardware());
                    println!("  MRENCLAVE:       {}...", &enclave.get_measurement().mrenclave[..16]);
                    println!("  MRSIGNER:        {}...", &enclave.get_measurement().mrsigner[..16]);
                }
                Err(e) => eprintln!("  Failed to init TEE: {}", e),
            }
        }
        "status" => {
            let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
            let status = enclave.status();
            let sealed = SealedStorage::new(SEALED_DIR);

            println!("\n  TEE Status");
            println!("  {}", "=".repeat(40));
            println!("  Enclave ID:      {}", &status.enclave_id[..8]);
            println!("  Backend:         {:?}", status.backend);
            println!("  Security Level:  {:?}", status.security_level);
            println!("  Healthy:         {}", status.is_healthy);
            println!("  Sealed Adapters: {}", sealed.count());
            println!("  {}", sealed.summary());
        }
        "seal" => {
            let prefix = match args.get(1) {
                Some(p) => p,
                None => {
                    eprintln!("Usage: dlrs tee seal <adapter-id-prefix>");
                    return;
                }
            };

            let mgr = load_manager();
            let matching: Vec<(String, String)> = mgr
                .list()
                .into_iter()
                .filter(|(id, _)| id.starts_with(prefix))
                .collect();

            if matching.is_empty() {
                eprintln!("  No adapter matching '{}'", prefix);
                return;
            }

            let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
            let mut sealed = SealedStorage::new(SEALED_DIR);

            for (id, _name) in &matching {
                let adapter = mgr.get(id).unwrap();
                let u = adapter.seed.lrim.u.as_slice().to_vec();
                let s = adapter.seed.lrim.sigma.as_slice().to_vec();
                let v = adapter.seed.lrim.v.as_slice().to_vec();

                match sealed.seal_adapter(
                    &mut enclave,
                    id,
                    &adapter.config.name,
                    adapter.config.domains.clone(),
                    adapter.config.rank,
                    adapter.config.original_dims,
                    &u,
                    &s,
                    &v,
                ) {
                    Ok(()) => println!(
                        "  Sealed [{}] '{}' into TEE ({:?})",
                        &id[..8],
                        adapter.config.name,
                        enclave.security_level()
                    ),
                    Err(e) => eprintln!("  Failed to seal [{}]: {}", &id[..8], e),
                }
            }
        }
        "attest" => {
            println!("\n  Running self-attestation...");
            let enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
            let quote = generate_quote(&enclave, "self-test-nonce");
            let policy = AttestationPolicy::default();
            let verdict = verify_quote(&quote, &policy);

            println!("  Backend:         {}", enclave.backend.name());
            println!("  MRENCLAVE:       {}...", &quote.measurement.mrenclave[..16]);
            println!("  Security Level:  {:?}", quote.security_level);
            println!("  Signature:       {}...", &quote.signature[..16]);
            println!("  Verdict:         {:?}", verdict);
        }
        "channel" => {
            println!("\n  TEE Secure Channel Demo...");
            println!("  {}", "-".repeat(50));

            let enclave_a = TeeEnclave::new(TeeBackend::Simulated).unwrap();
            let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();

            println!("  Enclave A: {}", &enclave_a.id[..8]);
            println!("  Enclave B: {}", &enclave_b.id[..8]);

            let policy = AttestationPolicy::default();
            let mut channel_a = SecureChannel::new(&enclave_a.id, policy.clone());
            let mut channel_b = SecureChannel::new(&enclave_b.id, policy);

            // Handshake
            let init = channel_a.initiate_handshake(&enclave_a);
            println!("  Step 1: A initiates handshake");

            let response = channel_b.respond_to_handshake(&enclave_b, &init);
            println!(
                "  Step 2: B responds (accepted={})",
                response.accepted
            );

            match channel_a.complete_handshake(&response, &init.dh_public) {
                Ok(()) => {
                    println!("  Step 3: A completes handshake");
                    println!("  Channel established!");
                    println!("  {}", channel_a.summary());

                    // Send a test message
                    let msg = channel_a
                        .encrypt_message(b"Hello from enclave A!")
                        .unwrap();
                    let decrypted = channel_b.decrypt_message(&msg).unwrap();
                    println!(
                        "  Test message: '{}' (encrypted + decrypted OK)",
                        String::from_utf8_lossy(&decrypted)
                    );
                }
                Err(e) => eprintln!("  Handshake failed: {}", e),
            }
        }
        other => {
            eprintln!("Unknown tee subcommand: {}", other);
            eprintln!("Usage: dlrs tee [init [sgx|trustzone|sim]|status|seal <id>|attest|channel]");
        }
    }
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
║              DLRS v1.0 — Full Demo                           ║
║       Training + Evolution + ZK Sharing + Backup             ║
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

    // Step 2: Local Training
    println!("\nStep 2: Local LoRA training on synthetic data...");
    println!("{}", "-".repeat(60));

    let dataset = Dataset::synthetic("demo-nlp", 256, 256, 30);
    let train_config = TrainConfig {
        epochs: 50,
        learning_rate: 0.01,
        loss_fn: LossFunction::MSE,
        lr_schedule: LrSchedule::CosineAnnealing { min_lr: 0.001 },
        patience: 15,
        min_delta: 1e-7,
        log_interval: 25,
    };

    let adapter = mgr.get_mut(&attn_q_id).unwrap();
    let history = train(adapter, &dataset, &train_config);
    println!(
        "  attn-query trained: best_loss={:.6} at epoch {} | final_loss={:.6}",
        history.best_loss, history.best_epoch, history.final_loss
    );

    let adapter = mgr.get_mut(&attn_v_id).unwrap();
    let history = train(adapter, &dataset, &train_config);
    println!(
        "  attn-value trained: best_loss={:.6} at epoch {} | final_loss={:.6}",
        history.best_loss, history.best_epoch, history.final_loss
    );

    // Step 3: AI-Assisted Auto-Training
    println!("\nStep 3: AI-assisted auto-training...");
    println!("{}", "-".repeat(60));

    let auto_dataset = Dataset::synthetic("auto-demo", 64, 64, 40);
    let (auto_adapter, auto_history, rec) = auto_train(
        "auto-tuned",
        "transformer.mlp",
        &auto_dataset,
        vec!["nlp".into(), "auto".into()],
    );
    println!("  Recommendation: {}", rec.explanation);
    println!(
        "  Auto-tuned result: rank={} | lr={:.5} | best_loss={:.6} | epochs={}",
        rec.rank, rec.learning_rate, auto_history.best_loss, auto_history.records.len()
    );
    mgr.add_adapter(auto_adapter);

    // Step 4: Evolution
    println!("\nStep 4: Further evolution...");
    println!("{}", "-".repeat(60));

    for (id, name) in [(&ffn_id, "ffn-up")] {
        let adapter = mgr.get(id).unwrap();
        let (m, n) = adapter.config.original_dims;
        let target = DMatrix::new_random(m, n) * 0.01;
        mgr.evolution_config.steps_per_cycle = 20;
        let fitness = mgr.evolve_adapter(id, &target).unwrap();
        println!(
            "  {} evolved -> fitness={:.4}, epoch={}",
            name, fitness, mgr.get(id).unwrap().seed.epoch
        );
    }

    // Step 5: ZK Capability Proofs
    println!("\nStep 5: Generating ZK capability proofs...");
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

    // Step 6: Merge adapters
    println!("\nStep 6: Merging attention adapters...");
    println!("{}", "-".repeat(60));

    let merged_id = mgr.merge_adapters(&attn_q_id, &attn_v_id).unwrap();
    println!("  {}", mgr.get(&merged_id).unwrap().summary());

    // Step 7: Apply to base weights
    println!("\nStep 7: Applying stacked LoRA to base weights...");
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

    // Step 8: Intelligent Sharing Registry
    println!("\nStep 8: Intelligent sharing registry...");
    println!("{}", "-".repeat(60));

    let mut registry = SharingRegistry::new();
    // Register some fake remote adapters to demonstrate discovery
    registry.register_remote(
        "remote-nlp-1".into(), "peer-abc".into(), "commit-xyz".into(),
        vec!["nlp".into(), "reasoning".into()], 0.85, 8,
    );
    registry.register_remote(
        "remote-vision-1".into(), "peer-def".into(), "commit-uvw".into(),
        vec!["vision".into()], 0.92, 16,
    );

    let nlp_candidates = registry.find_for_domain("nlp", 0.0);
    println!("  NLP candidates from network: {}", nlp_candidates.len());
    for r in &nlp_candidates {
        println!(
            "    {} | fitness={:.2} | reputation={:.2} | domains={:?}",
            &r.seed_id, r.fitness, r.reputation.overall, r.domains
        );
    }

    // Check which local adapters should be shared
    let shareable: Vec<_> = mgr.list().into_iter()
        .filter(|(id, _)| {
            mgr.get(id).map(|a| registry.should_share(a)).unwrap_or(false)
        })
        .collect();
    println!("  Local adapters eligible for sharing: {}", shareable.len());
    println!("  {}", registry.summary());

    // Step 9: Backup
    println!("\nStep 9: Creating versioned backup...");
    println!("{}", "-".repeat(60));

    let data = match mgr.to_json() {
        Ok(json) => json,
        Err(e) => {
            eprintln!("  Serialization error: {}", e);
            String::new()
        }
    };
    if !data.is_empty() {
        let mut backup = BackupManager::new(BACKUP_DIR);
        match backup.create_snapshot(&data, mgr.count(), "demo backup") {
            Ok(meta) => {
                println!(
                    "  Snapshot v{}: {} bytes, {} adapters, checksum={}...",
                    meta.version, meta.size_bytes, meta.adapter_count, &meta.checksum[..16]
                );
            }
            Err(e) => eprintln!("  Backup error: {}", e),
        }

        // Verify
        let results = backup.verify_all();
        let all_ok = results.iter().all(|(_, ok)| *ok);
        println!(
            "  Verification: {} snapshots, all intact: {}",
            results.len(),
            all_ok
        );
    }

    // Step 10: TEE Hardware Security
    println!("\nStep 10: TEE enclave operations...");
    println!("{}", "-".repeat(60));

    let mut enclave = TeeEnclave::new(TeeBackend::Simulated).unwrap();
    println!(
        "  Enclave: {} | backend={} | security={:?}",
        &enclave.id[..8],
        enclave.backend.name(),
        enclave.security_level()
    );

    // Seal adapter factors into enclave
    let mut sealed_storage = SealedStorage::new(SEALED_DIR);
    let adapter = mgr.get(&attn_q_id).unwrap();
    let u = adapter.seed.lrim.u.as_slice().to_vec();
    let s = adapter.seed.lrim.sigma.as_slice().to_vec();
    let v = adapter.seed.lrim.v.as_slice().to_vec();
    sealed_storage
        .seal_adapter(
            &mut enclave,
            &attn_q_id,
            "attn-query",
            vec!["nlp".into()],
            adapter.config.rank,
            adapter.config.original_dims,
            &u,
            &s,
            &v,
        )
        .unwrap();
    println!(
        "  Sealed attn-query factors into TEE ({} items)",
        enclave.status().sealed_items
    );

    // Compute proof inside enclave (factors never leave)
    let proof_result = enclave
        .enclave_compute_proof(&attn_q_id, "nlp", b"demo-challenge")
        .unwrap();
    println!(
        "  Enclave proof: score={:.4} | hash={}...",
        proof_result.capability_score,
        &proof_result.proof_hash[..16]
    );

    // Remote attestation
    let quote = generate_quote(&enclave, "demo-nonce");
    let verdict = verify_quote(&quote, &AttestationPolicy::default());
    println!(
        "  Attestation: MRENCLAVE={}... | verdict={:?}",
        &quote.measurement.mrenclave[..16],
        verdict
    );

    // Secure channel between two enclaves
    let enclave_b = TeeEnclave::new(TeeBackend::Simulated).unwrap();
    let policy = AttestationPolicy::default();
    let mut ch_a = SecureChannel::new(&enclave.id, policy.clone());
    let mut ch_b = SecureChannel::new(&enclave_b.id, policy);

    let init = ch_a.initiate_handshake(&enclave);
    let resp = ch_b.respond_to_handshake(&enclave_b, &init);
    ch_a.complete_handshake(&resp, &init.dh_public).unwrap();

    let msg = ch_a.encrypt_message(b"sealed adapter transfer").unwrap();
    let decrypted = ch_b.decrypt_message(&msg).unwrap();
    println!(
        "  Secure channel: {} <-> {} | msg OK ({} bytes)",
        &enclave.id[..8],
        &enclave_b.id[..8],
        decrypted.len()
    );

    // Step 11: Stats
    println!("\nStep 11: Final statistics...");
    println!("{}", "-".repeat(60));
    let stats = mgr.stats();
    println!("  Adapters:     {}", stats.total_adapters);
    println!("  Avg Fitness:  {:.4}", stats.avg_fitness);
    println!("  Best Fitness: {:.4}", stats.best_fitness);
    println!("  Domains:      {}", stats.total_domains);
    println!("  Parameters:   {}", stats.total_parameters);

    // Step 12: Persistence
    println!("\nStep 12: Saving state...");
    println!("{}", "-".repeat(60));
    save_manager(&mgr);

    // Step 13: P2P sharing
    println!("\nStep 13: Starting P2P node for ZK sharing...");
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
║              DLRS v1.0 Demo Complete!                        ║
║                                                              ║
║  - Created & trained 3 LoRA adapters (local + auto-tuning)  ║
║  - Generated ZK capability proofs (zero-knowledge)           ║
║  - Merged attention adapters, applied stacked LoRA           ║
║  - Intelligent sharing registry with reputation scoring      ║
║  - Versioned backup with SHA256 integrity verification       ║
║  - TEE enclave: sealed storage + attestation + secure channel║
║  - P2P auto-sharing via gossipsub + mDNS                     ║
║                                                              ║
║  Run 'dlrs serve' to start a persistent P2P node.            ║
║  Run 'dlrs auto-train' for AI-assisted fine-tuning.          ║
║  Run 'dlrs tee status' for TEE enclave information.          ║
╚══════════════════════════════════════════════════════════════╝
"#
    );
}
