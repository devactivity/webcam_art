




use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode}
};
use opencv::core::Size;
use opencv::{
    imgproc,
    videoio::{VideoCapture, CAP_ANY},
    prelude::*
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, Paragraph},
    Terminal
};

use std::{
    error::Error,
    io, panic,
    time::{Duration, Instant}
};

const ASCII_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
const TARGET_FPS: u64 = 30;

struct App {
    ascii_frame: String,
    fps: f64
}

impl App {
   fn new() -> Self {
        App { ascii_frame: String::new(), fps: 0.0 }
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

    // for y in (0..rows).step_by(6) {
    //     for x in (0..cols).step_by(6) {
    //         let pixel = gray.at_2d(y, x).unwrap();
    //         ascii_art.push(get_ascii_char(*pixel));
    //     }
    //
    //     ascii_art.push('\n');
    // }
    (0..rows)
        .step_by(1)
        .map(|y| {
            (0..cols)
                .step_by(1)
                .map(| x| {
                    let pixel = *gray.at_2d::<u8>(y, x).unwrap_or(&0);

                    get_ascii_char(pixel)
                })
                .collect::<String>()
        })
        .collect::<Vec<String>>()
        .join("\n")
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    camera: &mut VideoCapture
) -> io::Result<()> {
    let mut last_frame_time = Instant::now();
    let target_frame_time = Duration::from_micros(1_000_000 / TARGET_FPS);
    let mut frame = Mat::default();

    loop {
        let frame_start = Instant::now();

        // get terminal size
        let size = terminal.size()?;
        let term_width = size.width as usize;
        let term_height= size.height as usize;

        // process frame and update app
        camera.read(&mut frame).ok();

        if frame.empty() {
            continue;
        }

        // resize ACII based on terminal size
        let mut resized_frame = frame.clone();
        let (rows, cols) = (resized_frame.rows(), resized_frame.cols());
        let scale_x = cols as f64 / term_width as f64;
        let scale_y = rows as f64 / term_height as f64;

        // resize the frame to fit the terminal size
        imgproc::resize(
            &frame,
            &mut resized_frame,
            Size::new(term_width as i32, term_height as i32),
            scale_x,
            scale_y,
            imgproc::INTER_LINEAR
        ).unwrap();

        app.update(&resized_frame).ok();

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
                .split(f.area());

            let fps_text = format!("FPS: {:.2}", app.fps);
            let fps_paragraph = Paragraph::new(fps_text)
                .style(Style::default().fg(Color::Cyan))
                .block(Block::default().borders(Borders::ALL).title("Stats"));

            f.render_widget(fps_paragraph, chunks[0]);

            let ascii_paragraph = Paragraph::new(app.ascii_frame.as_str()).block(Block::default().borders(Borders::ALL).title("ASCII Webcam"));

            f.render_widget(ascii_paragraph, chunks[1]);
        })?;

        if event::poll(Duration::from_millis(1))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    return Ok(());
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
            std::thread::sleep(target_frame_time - processing_time);
        }
    }
}


fn reset_terminal() -> Result<(), Box<dyn Error>> {
    disable_raw_mode()?;

    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let original_hook=panic::take_hook();

    panic::set_hook(Box::new(move |panic_info| {
        reset_terminal().expect("failed to reset terminal");

        original_hook(panic_info);
    }));

    // setup terminal
    enable_raw_mode()?;

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // create app and run it 
    let mut camera = VideoCapture::new(0, CAP_ANY)?;
    let mut app = App::new();

    let res = run_app(&mut terminal, &mut app, &mut camera);

    // restore terminal
    reset_terminal()?;

    if let Err(err) = res {
        println!("Error: {:?}", err);
    }

    Ok(())
}
