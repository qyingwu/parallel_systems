#[macro_use]
extern crate log;
extern crate stderrlog;
extern crate clap;
extern crate ctrlc;
extern crate ipc_channel;
extern crate crossbeam_channel;
use std::env;
use std::fs;
use std::sync::Arc;
use coordinator::Coordinator;
use client::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::{Child,Command};
use ipc_channel::ipc::IpcSender as Sender;
use ipc_channel::ipc::IpcReceiver as Receiver;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::channel;
pub mod message;
pub mod oplog;
pub mod coordinator;
pub mod participant;
pub mod client;
pub mod checker;
pub mod tpcoptions;
use message::ProtocolMessage;
use participant::Participant; 


///
/// pub fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions) -> (std::process::Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     child_opts: CLI options for child process
///
/// 1. Set up IPC
/// 2. Spawn a child process using the child CLI options
/// 3. Do any required communication to set up the parent / child communication channels
/// 4. Return the child process handle and the communication channels for the parent
///
/// HINT: You can change the signature of the function if necessary
///
fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions) -> (Child, Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
    // Create IPC server with explicit type
    let (server, server_name) = IpcOneShotServer::<(Sender<ProtocolMessage>, Receiver<ProtocolMessage>)>::new().unwrap();
    child_opts.ipc_path = server_name.clone();

    // Spawn child process
    let child = Command::new(env::current_exe().unwrap())
        .args(child_opts.as_vec())
        .spawn()
        .expect("Failed to execute child process");

    // Accept connection from child and get their channels
    let (receiver, _) = server.accept().unwrap();
    let (child_tx, child_rx) = receiver.recv().unwrap();
    
    (child, child_tx, child_rx)
}





///
/// pub fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>)
///
///     opts: CLI options for this process
///
/// 1. Connect to the parent via IPC
/// 2. Do any required communication to set up the parent / child communication channels
/// 3. Return the communication channels for the child
///
/// HINT: You can change the signature of the function if necessasry
///
fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (Sender<ProtocolMessage>, Receiver<ProtocolMessage>) {
    // Increase initial delay and max attempts
    let mut retry_delay = 10; // Start with 100ms instead of 10ms
    let max_attempts = 5;     // Increase from 5 to 10 attempts
    
    for attempt in 0..max_attempts {
        info!("Attempting to connect to coordinator (attempt {}/{})", attempt + 1, max_attempts);
        
        match Sender::connect(opts.ipc_path.clone()) {
            Ok(coordinator_conn) => {
                // Create channels
                let (tx1, rx1) = channel().unwrap();
                let (tx2, rx2) = channel().unwrap();
                
                // Send channels to coordinator with error handling
                match coordinator_conn.send((tx1, rx2)) {
                    Ok(_) => {
                        info!("Successfully connected to coordinator");
                        return (tx2, rx1);
                    }
                    Err(e) => {
                        warn!("Failed to send channels to coordinator: {:?}", e);
                        if attempt < max_attempts - 1 {
                            std::thread::sleep(std::time::Duration::from_millis(retry_delay));
                            retry_delay *= 2;
                            continue;
                        }
                    }
                }
            }
            Err(e) => {
                warn!("Connection attempt {} failed: {:?}", attempt + 1, e);
                if attempt < max_attempts - 1 {
                    std::thread::sleep(std::time::Duration::from_millis(retry_delay));
                    retry_delay *= 2;
                    continue;
                }
            }
        }
    }
    
    panic!("Failed to establish connection with coordinator after {} attempts. Check if coordinator is running and the IPC path is correct: {:?}", max_attempts, opts.ipc_path);
}




///
/// pub fn run(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Creates a new coordinator
/// 2. Spawns and connects to new clients processes and then registers them with
///    the coordinator
/// 3. Spawns and connects to new participant processes and then registers them
///    with the coordinator
/// 4. Starts the coordinator protocol
/// 5. Wait until the children finish execution
///
fn run(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    // Create a unique IPC path
    let ipc_path = format!("/tmp/tpc_coordinator_{}", std::process::id());
    info!("Setting up coordinator with IPC path: {}", ipc_path);

    // Ensure log directory exists
    let log_dir = std::path::Path::new(&opts.log_path);
    std::fs::create_dir_all(log_dir).unwrap_or_else(|e| {
        panic!("Failed to create log directory {:?}: {}", log_dir, e)
    });

    let coord_log_path = log_dir.join("coordinator.log");
    let running_flag = running.clone();
    
    // Create coordinator first
    info!("Creating coordinator...");
    let mut coordinator = Coordinator::new(coord_log_path.to_str().unwrap().to_string(), &running_flag);
    
    info!("Spawning {} participants...", opts.num_participants);
    let mut participant_children = Vec::new();
    
    // Wait for coordinator to initialize
    std::thread::sleep(std::time::Duration::from_millis(500));

    // Spawn participants
    for i in 0..opts.num_participants {
        let participant_id = format!("participant_{}", i);
        info!("Spawning participant {}", participant_id);
        
        let (tx_to_coord, rx_from_part) = channel().unwrap();
        let (tx_to_part, rx_from_coord) = channel().unwrap();
        
        // Register participant with coordinator first
        coordinator.participant_join(i as u32, tx_to_part.clone(), rx_from_part);
        
        // Spawn participant process
        let participant_log_path = log_dir.join(format!("participant_{}.log", i));
        let child = spawn_participant(
            i as u32,
            participant_log_path.to_str().unwrap(),
            tx_to_coord,
            rx_from_coord,
            running.clone(),
        );
        
        participant_children.push(child);
        info!("Participant {} spawned successfully", participant_id);
    }

    // Start coordinator protocol
    info!("Starting coordinator protocol...");
    coordinator.protocol();

    // Wait for all participants to finish
    for child in participant_children {
        let _ = child.wait_with_output();
    }
}

fn spawn_participant(
    id: u32,
    log_path: &str,
    tx: Sender<ProtocolMessage>,
    rx: Receiver<ProtocolMessage>,
    running: Arc<AtomicBool>,
) -> std::process::Child {
    info!("Spawning participant {} with log path: {}", id, log_path);
    
    let mut participant = Participant::new(
        format!("participant_{}", id),
        log_path.to_string(),
        running,
        0.95,  // send_success_prob
        0.95,  // operation_success_prob
        rx,    // This is from_coordinator (Receiver)
        tx,    // This is to_coordinator (Sender)
    );

    std::thread::spawn(move || {
        info!("Starting participant {} protocol", id);
        participant.protocol();
    });

    std::process::Command::new(std::env::current_exe().unwrap())
        .arg("--mode")
        .arg("participant")
        .arg("--id")
        .arg(id.to_string())
        .arg("--log-path")
        .arg(log_path)
        .spawn()
        .expect("Failed to spawn participant process")
}


///
/// pub fn run_client(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>, shutdown_rx: crossbeam_channel::Receiver<()>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new client
/// 3. Starts the client protocol
///
fn run_client(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    let client_id_str = format!("client_{}", opts.num);
    //let running_flag = Arc::new(AtomicBool::new(true));

    // Create and start client
    let (to_coordinator, from_coordinator) = connect_to_coordinator(opts);

    let mut client = Client::new(
        client_id_str,
        running.clone(),
        to_coordinator,
        from_coordinator,
    );

    // Start the client
    client.start();

}


///
/// pub fn run_participant(opts: &tpcoptions:TPCOptions, running: Arc<AtomicBool>, shutdown_rx: crossbeam_channel::Receiver<()>)
///     opts: An options structure containing the CLI arguments
///     running: An atomically reference counted (ARC) AtomicBool(ean) that is
///         set to be false whenever Ctrl+C is pressed
///
/// 1. Connects to the coordinator to get tx/rx
/// 2. Constructs a new participant
/// 3. Starts the participant protocol
///
fn run_participant(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    // Ensure log directory exists
    fs::create_dir_all(&opts.log_path)
        .unwrap_or_else(|e| panic!("Failed to create log directory {}: {}", opts.log_path, e));

    let participant_id_str = format!("participant_{}", opts.num);
    let participant_log_path = format!("{}/participant_{}.log", opts.log_path, opts.num);
    
    // Connect to coordinator
    let (to_coordinator, from_coordinator) = connect_to_coordinator(opts);

    let mut participant = participant::Participant::new(
        participant_id_str,
        participant_log_path,
        running.clone(),
        0.95,  // send_success_prob
        0.95,  // operation_success_prob
        from_coordinator,
        to_coordinator,
    );

    // Start the participant's protocol
    participant.start();

}


fn main() {
    // Parse CLI arguments
    let opts = tpcoptions::TPCOptions::new();

    // Set up logging
    stderrlog::new()
        .module(module_path!())
        .quiet(false)
        .timestamp(stderrlog::Timestamp::Millisecond)
        .verbosity(opts.verbosity)
        .init()
        .unwrap();

    match fs::create_dir_all(opts.log_path.clone()) {
        Err(e) => error!("Failed to create log_path: \"{:?}\". Error \"{:?}\"", opts.log_path, e),
        _ => (),
    }
    

     // Set up Ctrl-C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    let m = opts.mode.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        if m == "run" {
            print!("\n");
        }
    }).expect("Error setting signal handler!");


    // Execute main logic
    match opts.mode.as_ref() {
        "run" => run(&opts, running),
        "client" => run_client(&opts, running),
        "participant" => run_participant(&opts, running),
        "check" => checker::check_last_run(
            opts.num_clients,
            opts.num_requests,
            opts.num_participants,
            &opts.log_path,
        ),
        _ => panic!("Unknown mode: {}", opts.mode),
    }
}

