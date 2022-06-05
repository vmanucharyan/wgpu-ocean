#![allow(dead_code)]

mod camera;
mod generate_plane;
mod ocean;
mod renderer;
mod vertex;

use winit::dpi::LogicalSize;
use winit::{
  event::*,
  event::{Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::WindowBuilder,
};

use renderer::Renderer;

fn main() {
  pollster::block_on(run());
}

pub async fn run() {
  use std::time::Instant;

  env_logger::init();

  let event_loop = EventLoop::new();
  let window = WindowBuilder::new()
    .with_inner_size(LogicalSize::new(1280.0, 800.0))
    .build(&event_loop)
    .unwrap();

  let mut state = Renderer::new(&window).await;
  let mut is_focused = false;

  let start_instant = Instant::now();
  let mut last_frame_instant = Instant::now();

  event_loop.run(move |event, _, control_flow| match event {
    Event::DeviceEvent {
      event: DeviceEvent::MouseMotion{ delta, },
      .. // We're not using device_id currently
  } => if state.mouse_pressed {
      state.camera_controller.process_mouse(delta.0, delta.1)
  }
    Event::WindowEvent {
      ref event,
      window_id,
    } if window_id == window.id() => {
      if !state.input(event) {
        match event {
          WindowEvent::CloseRequested
          | WindowEvent::KeyboardInput {
            input:
              KeyboardInput {
                state: ElementState::Pressed,
                virtual_keycode: Some(VirtualKeyCode::Escape),
                ..
              },
            ..
          } => *control_flow = ControlFlow::Exit,

          WindowEvent::Resized(physical_size) => {
            state.resize(*physical_size);
          }

          WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
            // new_inner_size is &&mut so we have to dereference it twice
            state.resize(**new_inner_size);
          }

          WindowEvent::Focused(value) => is_focused = *value,

          _ => {}
        }
      }
    }

    Event::RedrawRequested(window_id) if window_id == window.id() => {
      let now = Instant::now();
      let dt = now - last_frame_instant;
      last_frame_instant = now;
      state.update(dt);

      let time = start_instant.elapsed().as_secs_f32();
      match state.render(time, dt) {
        Ok(_) => {}
        // Reconfigure the surface if lost
        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
        // The system is out of memory, we should probably quit
        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
        // All other errors (Outdated, Timeout) should be resolved by the next frame
        Err(e) => eprintln!("{:?}", e),
      }
    }

    Event::MainEventsCleared => {
      // RedrawRequested will only trigger once, unless we manually
      // request it.
      let visible = window.is_visible().unwrap_or(true);
      if visible && is_focused {
        window.request_redraw();
      }
    }

    _ => {}
  })
}
