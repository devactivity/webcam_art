use color_eyre::{eyre, Result};
use env_logger::Builder;
use log::LevelFilter;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use opus::{Channels, Encoder as OpusEncoder};
use parking_lot::Mutex;
use ringbuf::HeapRb;

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use opencv::{
    core::Size,
    imgproc,
    prelude::*,
    videoio::{VideoCapture, VideoWriter, CAP_ANY},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use std::{
    env,
    fs::File,
    io::{self, Write},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const ASCII_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
const TARGET_FPS: u64 = 30;
const FPS: f64 = 30.0;
const SAMPLE_RATE: u32 = 48000;
const CHANNELS: u16 = 1;

struct App {
    ascii_frame: String,
    fps: f64,
    is_recording: bool,
}

impl App {
    fn new() -> Self {
        App {
            ascii_frame: String::new(),
            fps: 0.0,
            is_recording: false,
        }
    }

    fn update(&mut self, frame: &Mat) -> opencv::Result<()> {
        self.ascii_frame = process_frame(frame);

        Ok(())
    }
}

fn get_ascii_char(value: u8) -> char {
    let index = (value as usize * (ASCII_CHARS.len() - 1)) / 255;

    if index < ASCII_CHARS.len() {
        ASCII_CHARS[index]
    } else {
        ASCII_CHARS[ASCII_CHARS.len() - 1]
    }
}

fn process_frame(frame: &Mat) -> String {
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

    let (rows, cols) = (gray.rows(), gray.cols());

    if rows == 0 || cols == 0 {
        return String::new();
    }

    (0..rows)
        .step_by(1)
        .map(|y| {
            (0..cols)
                .step_by(1)
                .map(|x| {
                    let pixel = *gray.at_2d::<u8>(y, x).unwrap_or(&0);

                    get_ascii_char(pixel)
                })
                .collect::<String>()
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn setup_logging() -> Result<(), Box<dyn std::error::Error>> {
    let log_file = File::create("output.log")?;

    Builder::new()
        .filter(None, LevelFilter::Info)
        .format(|buf, record| writeln!(buf, "{}: {}", record.level(), record.args()))
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    Ok(())
}

fn render_ascii_frame(ascii: &str, size: Size) -> Result<Mat> {
    let mut img =
        Mat::new_size_with_default(size, opencv::core::CV_8UC3, opencv::core::Scalar::all(0.0))?;
    let font = opencv::imgproc::FONT_HERSHEY_PLAIN;
    let font_scale = 0.4;
    let thickness = 1;
    let color = opencv::core::Scalar::new(255.0, 255.0, 255.0, 0.0);

    for (i, line) in ascii.lines().enumerate() {
        imgproc::put_text(
            &mut img,
            line,
            opencv::core::Point::new(0, (i as i32 + 1) * 10),
            font,
            font_scale,
            color,
            thickness,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(img)
}

fn start_recording(camera: Arc<Mutex<VideoCapture>>, is_recording: Arc<AtomicBool>) -> Result<()> {
    let current_dir = env::current_dir()?;
    let output_video = current_dir.join("output_ascii.mp4");
    let output_audio = current_dir.join("output.opus");
    let final_output = current_dir.join("final_output.mp4");
    let frame_size = Size::new(640, 480);

    let fourcc = VideoWriter::fourcc('a', 'v', 'c', '1')?;
    let mut video_writer = VideoWriter::new(
        output_video.to_str().unwrap(),
        fourcc,
        FPS,
        frame_size,
        true,
    )?;

    // initialize audio recording
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| eyre::eyre!("no input device available"))?;
    let config = cpal::StreamConfig {
        channels: CHANNELS,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let mut opus_encoder = OpusEncoder::new(SAMPLE_RATE, Channels::Mono, opus::Application::Audio)?;
    let ring_buffer = HeapRb::<f32>::new(SAMPLE_RATE as usize * CHANNELS as usize);
    let (mut producer, mut consumer) = ring_buffer.split();

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            producer.push_slice(data);
        },
        |err| log::info!("an error occurred on the audio input stream: {}", err),
        None,
    )?;

    stream.play()?;

    // process audio file
    let mut audio_file = File::create(&output_audio)?;
    let mut frame_buffer = vec![0.0f32; SAMPLE_RATE as usize / 50]; // 20ms frame

    while is_recording.load(Ordering::Relaxed) {
        let mut frame = Mat::default();
        {
            let mut camera = camera.lock();
            camera.read(&mut frame)?;
        }

        if !frame.empty() {
            let ascii_frame = process_frame(&frame);
            let ascii_image = render_ascii_frame(&ascii_frame, frame_size)?;

            video_writer.write(&ascii_image)?;
        }

        // process audio
        while consumer.len() >= frame_buffer.len() {
            consumer.pop_slice(&mut frame_buffer);

            let mut output_buffer = vec![0_u8; 1275];
            let bytes_encoded = opus_encoder.encode_float(&frame_buffer, &mut output_buffer)?;

            audio_file.write_all(&output_buffer[..bytes_encoded])?;
        }

        thread::sleep(Duration::from_millis((1000.0 / FPS) as u64));
    }

    // stop audio recording
    drop(stream);

    // combine video and audio
    let output = Command::new("ffmpeg")
        .args([
            "-i",
            output_video.to_str().unwrap(),
            "-i",
            output_audio.to_str().unwrap(),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            final_output.to_str().unwrap(),
        ])
        .output()?;

    if !output.status.success() {
        log::info!("ffmpeg error: {}", String::from_utf8_lossy(&output.stderr));
        return Err(eyre::eyre!("ffmpeg command failed"));
    }

    // clean up temporary files
    std::fs::remove_file(&output_video)?;
    std::fs::remove_file(&output_audio)?;

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    camera: Arc<Mutex<VideoCapture>>,
) -> io::Result<()> {
    let mut last_frame_time = Instant::now();
    let target_frame_time = Duration::from_micros(1_000_000 / TARGET_FPS);
    let is_recording = Arc::new(AtomicBool::new(false));
    let recording_thread: Arc<Mutex<Option<JoinHandle<Result<()>>>>> = Arc::new(Mutex::new(None));

    loop {
        let frame_start = Instant::now();

        // get terminal size
        let size = terminal.size()?;
        let term_width = size.width as i32;
        let term_height = size.height as i32;

        // process frame and update app
        let mut frame = Mat::default();
        {
            let mut camera = camera.lock();
            camera
                .read(&mut frame)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        }

        if !frame.empty() {
            let mut resized_frame = Mat::default();

            imgproc::resize(
                &frame,
                &mut resized_frame,
                Size::new(term_width, term_height),
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

            app.update(&resized_frame)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        }

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
                .split(f.area());

            let status = if app.is_recording {
                "Recording"
            } else {
                "Not Recording"
            };
            let stats_text = format!("FPS: {:.2} | Status: {}", app.fps, status);
            let stats_paragraph = Paragraph::new(stats_text)
                .style(Style::default().fg(Color::Cyan))
                .block(Block::default().borders(Borders::ALL).title("Stats"));

            f.render_widget(stats_paragraph, chunks[0]);

            let ascii_paragraph = Paragraph::new(app.ascii_frame.as_str())
                .block(Block::default().borders(Borders::ALL).title("ASCII Webcam"));

            f.render_widget(ascii_paragraph, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => {
                        // stop recording if it's ongoing
                        if app.is_recording {
                            is_recording.store(false, Ordering::Relaxed);

                            if let Some(handle) = recording_thread.lock().take() {
                                match handle.join() {
                                    Ok(result) => {
                                        if let Err(e) = result {
                                            log::info!("recording error: {:?}", e);
                                        }
                                    }
                                    Err(e) => {
                                        log::info!("failed to join recording thread: {:?}", e)
                                    }
                                }
                            }
                        }

                        return Ok(());
                    }
                    KeyCode::Char('r') => {
                        if !app.is_recording {
                            app.is_recording = true;

                            is_recording.store(true, Ordering::Relaxed);
                            let camera_clone = camera.clone();
                            let is_recording_clone = is_recording.clone();
                            let handle = thread::spawn(move || {
                                start_recording(camera_clone, is_recording_clone)
                            });

                            *recording_thread.lock() = Some(handle);
                        } else {
                            app.is_recording = false;

                            is_recording.store(false, Ordering::Relaxed);

                            if let Some(handle) = recording_thread.lock().take() {
                                match handle.join() {
                                    Ok(result) => {
                                        if let Err(e) = result {
                                            log::info!("recording error: {:?}", e);
                                        }
                                    }
                                    Err(e) => {
                                        log::info!("failed to join recording thread: {:?}", e)
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let current_frame_time = Instant::now();

        app.fps = 1.0
            / current_frame_time
                .duration_since(last_frame_time)
                .as_secs_f64();
        last_frame_time = current_frame_time;

        let processing_time = frame_start.elapsed();
        if processing_time < target_frame_time {
            thread::sleep(target_frame_time - processing_time);
        }
    }
}

fn reset_terminal() -> Result<()> {
    disable_raw_mode()?;

    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;

    Ok(())
}

fn main() -> Result<()> {
    color_eyre::install()?;
    setup_logging().ok();

    // let original_hook = panic::take_hook();
    // panic::set_hook(Box::new(move |panic_info| {
    //     reset_terminal().expect("failed to reset terminal");
    //
    //     original_hook(panic_info);
    // }));

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // initialize camera
    let camera = Arc::new(Mutex::new(VideoCapture::new(0, CAP_ANY)?));
    let mut app = App::new();

    let res = run_app(&mut terminal, &mut app, camera.clone());

    // restore terminal
    reset_terminal()?;

    if let Err(err) = res {
        println!("Error: {:?}", err);
    }

    Ok(())
}
