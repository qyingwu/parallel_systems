#[macro_use]
extern crate log;
extern crate stderrlog;
extern crate clap;
extern crate ctrlc;
extern crate ipc_channel;
extern crate crossbeam_channel;
use std::env;
use std::fs;
use std::sync::{Arc, Barrier};
use std::sync::Mutex;
use coordinator::Coordinator;
use client::Client;
use std::sync::atomic::{AtomicBool, Ordering};
use std::process::{Child, Command, Stdio};
use ipc_channel::ipc::{IpcSender, IpcReceiver, IpcOneShotServer, channel};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;
pub mod message;
pub mod oplog;
pub mod coordinator;
pub mod participant;
pub mod client;
pub mod checker;
pub mod tpcoptions;
use message::{MessageType, ProtocolMessage};
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
fn spawn_child_and_connect(child_opts: &mut tpcoptions::TPCOptions) -> (Child, IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>) {
    // Step 1: Create IPC server with explicit type
    let (server, server_name) = IpcOneShotServer::<(IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>)>::new()
        .unwrap_or_else(|e| {
            error!("Failed to create IPC server: {:?}", e);
            panic!("IPC server creation failed");
        });


    // Step 2: Set `ipc_path` for the child options
    child_opts.ipc_path = server_name.clone();

    // Step 3: Log the executable path
    let exe_path = env::current_exe().unwrap();
    // Step 4: Spawn child process
    let child = Command::new(exe_path)
        .args(child_opts.as_vec())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| {
            error!("Failed to spawn child process: {:?}", e);
            panic!("Child process spawn failed");
        });


    // Step 5: Wait for the child process to connect
    let (_receiver, (sender, receiver)) = server.accept().unwrap_or_else(|e| {
        error!("Failed to accept connection from child: {:?}", e);
        panic!("Child connection acceptance failed");
    });

    // Step 7: Return child handle and channels
    (child, sender, receiver)
}





///
/// pub fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>)
///
///     opts: CLI options for this process
///
/// 1. Connect to the parent via IPC
/// 2. Do any required communication to set up the parent / child communication channels
/// 3. Return the communication channels for the child
///
/// HINT: You can change the signature of the function if necessasry
///
fn connect_to_coordinator(opts: &tpcoptions::TPCOptions) -> (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>) {
    let (to_child, from_parent): (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>) = channel().unwrap();
    let (to_parent, from_child): (IpcSender<ProtocolMessage>, IpcReceiver<ProtocolMessage>) = channel().unwrap();

    // Connect to the coordinator's IPC server
    let tx = IpcSender::connect(opts.ipc_path.clone())
        .unwrap_or_else(|e| {
            println!("ERROR: Failed to connect to coordinator: {:?}", e);
            error!("Failed to connect to coordinator: {:?}", e);
            panic!("Connection failed");
        });
    
    tx.send((to_child, from_child)).unwrap();

    (to_parent, from_parent)
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

     // Step 1: Create Coordinator
     //Arc: Atomic Reference Counting
     //Thread-Safe Shared Ownership
     // Coordinator must be accessible to multiple threads
    let coordinator = Arc::new(Mutex::new(Coordinator::new(
        format!("{}/coordinator.log", opts.log_path),
        &running,
    )));

    //Step 2 clone coordinator
    // Create a Barrier to Synchronize Initialization
    //Two threads: main thread and coordinator thread
    let barrier = Arc::new(Barrier::new(2)); 
    let barrier_clone = barrier.clone();
    let coordinator_clone = coordinator.clone();
    
    //step 3: Create and Connect Clients
    let mut handles: Vec<Child> = vec![];
    //Iterates for Each Client:
    for i in 0..opts.num_clients {
        let mut client_opts = opts.clone();
        client_opts.mode = "client".to_string();
        client_opts.num = i;
        //Use spawn to start a child process for the client
        // establish IPC channels (client_tx and client_rx) for communication with the coordinator.
        let (child, client_tx, client_rx) = spawn_child_and_connect(&mut client_opts);
        
        //Registers the Client with the Coordinator:
        coordinator.lock().unwrap().client_join(i, client_tx, client_rx);
        //Stores the Client Process Handle
        handles.push(child);
    }

    // Step 4: Create and Connect Participants
    for i in 0..opts.num_participants {
        let mut participant_opts = opts.clone();
        participant_opts.mode = "participant".to_string();
        participant_opts.num = i;

        let (child, participant_tx, participant_rx) = spawn_child_and_connect(&mut participant_opts);
        
        coordinator.lock().unwrap().participant_join(i, participant_tx, participant_rx);
        handles.push(child);
    }

    // Start the Coordinator in a Separate Thread
    let coord_handle = thread::spawn(move || {
        let mut coord = coordinator_clone.lock().unwrap();
        coord.start(); // Coordinator's main logic
        barrier_clone.wait(); // Signal main thread that initialization is complete
    });

    // Wait for the Coordinator Thread to Signal Initialization
    barrier.wait(); 

    // Step 5: Wait for All Child Processes to Finish
    for mut handle in handles {
        handle.wait().expect("Child process failed");
    }

    // Step 6: Wait for Coordinator Thread to Complete
    coord_handle.join().expect("Coordinator thread failed");

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

    // Create and start client
    let (to_coordinator, from_coordinator) = connect_to_coordinator(opts);

    let mut client = Client::new(
        client_id_str,
        running.clone(),
        opts.num_requests,  
        to_coordinator,
        from_coordinator,
    );

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
        opts.send_success_probability,
        opts.operation_success_probability,
        from_coordinator,
        to_coordinator,
    );

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

