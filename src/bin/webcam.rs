use crossterm::{
    execute,
    terminal::{Clear, ClearType},
};
use opencv::{
    imgproc,
    prelude::*,
    videoio::{VideoCapture, CAP_ANY},
};
use std::{
    io::{self, Write},
    time::Duration,
};

const ASCII_CHARS: &[char] = &[' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

fn get_ascii_char(value: u8) -> char {
    let index = (value as usize * (ASCII_CHARS.len() - 1)) / 255;

    ASCII_CHARS[index]
}

fn process_frame(frame: Mat) -> String {
    let mut ascii_art = String::new();
    let mut gray = Mat::default();

    imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();

    let (rows, cols) = (gray.rows(), gray.cols());

    for y in (0..rows).step_by(6) {
        for x in (0..cols).step_by(6) {
            let pixel = gray.at_2d(y, x).unwrap();
            ascii_art.push(get_ascii_char(*pixel));
        }

        ascii_art.push('\n');
    }

    ascii_art
}

fn main() -> opencv::Result<()> {
    let mut camera = VideoCapture::new(0, CAP_ANY)?;
    let mut frame = Mat::default();

    loop {
        camera.read(&mut frame)?;

        if frame.empty() {
            continue;
        }

        let ascii_art = process_frame(frame.clone());

        match execute!(io::stdout(), Clear(ClearType::All)) {
            Ok(()) => {
                print!("{}", ascii_art);
                let _ = io::stdout().flush();
            }
            Err(e) => {
                eprintln!("failed to clear temrinal: {}", e);
                return Err(opencv::Error::new(
                    opencv::core::StsError,
                    "error clearing terminal",
                ));
            }
        }

        print!("{}", ascii_art);
        let _ = io::stdout().flush();

        std::thread::sleep(Duration::from_millis(100));
    }
}
