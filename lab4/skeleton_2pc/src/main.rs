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
use std::sync::Mutex;
use coordinator::Coordinator;
use client::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::{Child, Command, Stdio};
use ipc_channel::ipc::IpcSender as Sender;
use ipc_channel::ipc::IpcReceiver as Receiver;
use ipc_channel::ipc::IpcOneShotServer;
use ipc_channel::ipc::channel;
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;
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
    info!("Starting spawn_child_and_connect with options: {:?}", child_opts);
    
    // Create IPC server with explicit type
    let (server, server_name) = IpcOneShotServer::<(Sender<ProtocolMessage>, Receiver<ProtocolMessage>)>::new()
        .unwrap_or_else(|e| {
            error!("Failed to create IPC server: {:?}", e);
            panic!("IPC server creation failed");
        });
    
    info!("Created IPC server with name: {}", server_name);
    child_opts.ipc_path = server_name.clone();

    // Spawn child process
    info!("Spawning child process with args: {:?}", child_opts.as_vec());
    let child = Command::new(env::current_exe().unwrap())
        .args(child_opts.as_vec())
        .spawn()
        .unwrap_or_else(|e| {
            error!("Failed to spawn child process: {:?}", e);
            panic!("Child process spawn failed");
        });
    
    info!("Child process spawned with PID: {}", child.id());

    // Accept connection from child
    info!("Waiting for child process to connect...");
    let (receiver, _) = server.accept().unwrap_or_else(|e| {
        error!("Failed to accept connection from child: {:?}", e);
        panic!("Child connection acceptance failed");
    });
    info!("Accepted connection from child process");

    // Get channels from child
    info!("Waiting to receive channels from child...");
    let (child_tx, child_rx) = receiver.recv().unwrap_or_else(|e| {
        error!("Failed to receive channels from child: {:?}", e);
        panic!("Channel reception failed");
    });
    info!("Successfully received channels from child");
    
    info!("Child process setup completed successfully");
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
    println!("Starting connect_to_coordinator with IPC path: {:?}", opts.ipc_path);
    
    let mut retry_delay = 10;
    let max_attempts = 5;
    
    for attempt in 0..max_attempts {
        println!("=== Connection Attempt {}/{} ===", attempt + 1, max_attempts);
        println!("Current retry delay: {}ms", retry_delay);
        
        match Sender::connect(opts.ipc_path.clone()) {
            Ok(coordinator_conn) => {
                println!("✓ Successfully established initial connection to coordinator");
                
                // Create channels
                println!("Creating IPC channels...");
                let (tx1, rx1) = match channel() {
                    Ok(channels) => {
                        println!("✓ Successfully created first channel pair");
                        channels
                    },
                    Err(e) => {
                        println!("✗ Failed to create first channel pair: {:?}", e);
                        if attempt < max_attempts - 1 {
                            sleep(Duration::from_millis(retry_delay));
                            retry_delay *= 2;
                            continue;
                        } else {
                            panic!("Failed to create channels after all attempts");
                        }
                    }
                };
                
                let (tx2, rx2) = match channel() {
                    Ok(channels) => {
                        println!("✓ Successfully created second channel pair");
                        channels
                    },
                    Err(e) => {
                        println!("✗ Failed to create second channel pair: {:?}", e);
                        if attempt < max_attempts - 1 {
                            sleep(Duration::from_millis(retry_delay));
                            retry_delay *= 2;
                            continue;
                        } else {
                            panic!("Failed to create channels after all attempts");
                        }
                    }
                };
                
                // Send channels to coordinator
                println!("Sending channels to coordinator...");
                match coordinator_conn.send((tx1, rx2)) {
                    Ok(_) => {
                        println!("✓ Successfully sent channels to coordinator");
                        println!("Connection process completed successfully!");
                        return (tx2, rx1);
                    }
                    Err(e) => {
                        println!("✗ Failed to send channels to coordinator: {:?}", e);
                        if attempt < max_attempts - 1 {
                            sleep(Duration::from_millis(retry_delay));
                            retry_delay *= 2;
                            continue;
                        }
                    }
                }
            }
            Err(e) => {
                println!("✗ Initial connection attempt {} failed: {:?}", attempt + 1, e);
                if attempt < max_attempts - 1 {
                    println!("Waiting {}ms before next attempt...", retry_delay);
                    sleep(Duration::from_millis(retry_delay));
                    retry_delay *= 2;
                    continue;
                }
            }
        }
    }
    
    let error_msg = format!("Failed to establish connection with coordinator after {} attempts. Check if coordinator is running and the IPC path is correct: {:?}", max_attempts, opts.ipc_path);
    println!("✗ {}", error_msg);
    panic!("{}", error_msg);
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
enum ProcessHandle {
    Thread(JoinHandle<()>),
    Process(Child)
}

fn run(opts: &tpcoptions::TPCOptions, running: Arc<AtomicBool>) {
    println!("Starting run function with options: {:?}", opts);
    
    // Create log directory
    let log_dir = std::path::Path::new(&opts.log_path);
    println!("Creating log directory at: {:?}", log_dir);
    std::fs::create_dir_all(log_dir).unwrap_or_else(|e| {
        println!("ERROR: Failed to create log directory: {}", e);
        panic!("Failed to create log directory {:?}: {}", log_dir, e)
    });

    let coord_log_path = log_dir.join("coordinator.log");
    println!("Coordinator log path: {:?}", coord_log_path);

    // Create coordinator
    println!("Creating coordinator...");
    let coordinator = Arc::new(Mutex::new(
        Coordinator::new(coord_log_path.to_str().unwrap().to_string(), &running)
    ));
   
    // Then declare your handles vector as:
    let mut handles: Vec<ProcessHandle> = Vec::new();

    // Create and start coordinator thread
    let coordinator_clone = coordinator.clone();  // Clone Arc<Mutex<Coordinator>>
    let running_clone = running.clone();          // Clone Arc<AtomicBool>

    let coord_handle = thread::spawn(move || {
        let mut coord = coordinator_clone.lock().unwrap();
        coord.start();  // or whatever method starts your coordinator's main loop
    });

    // For thread handles:
    handles.push(ProcessHandle::Thread(coord_handle));

    // Wait for coordinator to initialize
    sleep(Duration::from_millis(1000));

    // Create clients
    println!("Starting client creation. Number of clients: {}", opts.num_clients);
    for i in 0..opts.num_clients {
        let client_id = i as u32;
        let client_log_path = log_dir.join(format!("client_{}.log", client_id));

        println!("Spawning client {} process", client_id);
        let client_handle = Command::new(env::current_exe().unwrap())
            .arg("-m")
            .arg("client")
            .arg("--ipc_path")
            .arg(&opts.ipc_path)
            .arg("-l")
            .arg(&opts.log_path)
            .arg("--num")
            .arg(&client_id.to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to spawn client process");
        
        handles.push(ProcessHandle::Process(client_handle));    
        println!("Client {} process spawned", client_id);
        sleep(Duration::from_millis(100));
    }

    // Create participants
    println!("Starting participant creation. Number of participants: {}", opts.num_participants);
    for i in 0..opts.num_participants {
        let participant_id = i as u32;
        let participant_log_path = log_dir.join(format!("participant_{}.log", participant_id));

        println!("Spawning participant {} process", participant_id);
        let participant_handle = Command::new(env::current_exe().unwrap())
            .arg("-m")
            .arg("participant")
            .arg("--ipc_path")
            .arg(&opts.ipc_path)
            .arg("-l")
            .arg(&opts.log_path)
            .arg("--num")
            .arg(&participant_id.to_string())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to spawn participant process");
        handles.push(ProcessHandle::Process(participant_handle));    
        println!("Participant {} process spawned", participant_id);
        sleep(Duration::from_millis(100));
    }
    println!("All participants created and initialized");

    // Wait for all processes to complete

    for (i, handle) in handles.into_iter().enumerate() {
        println!("Waiting for process/thread {} to complete", i);
        match handle {
            ProcessHandle::Thread(h) => {
                let _ = h.join().unwrap();
            },
            ProcessHandle::Process(mut h) => {
                let _ = h.wait().unwrap();
            }
        }
        println!("Process/thread {} completed", i);
    }
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
    

    // Set-up Ctrl-C / SIGINT handler
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

