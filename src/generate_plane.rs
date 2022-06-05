use crate::vertex::Vertex;

pub fn generate_plane(size: f32, num_vertices: usize) -> (Vec<Vertex>, Vec<u32>) {
  let num_vertices_normalized = if num_vertices < 2 { 2 } else { num_vertices };

  let step = size / ((num_vertices_normalized - 1) as f32);
  let mut vertices = vec![(0.0f32, 0.0f32); num_vertices_normalized * num_vertices_normalized];
  let mut indices: Vec<u32> = Vec::new();

  for x_row in 0..num_vertices_normalized - 1 {
    let y_row = 0;

    let i0 = x_row + (num_vertices * y_row);
    let x0 = -size * 0.5 + (x_row as f32) * step;
    let y0 = -size * 0.5 + (y_row as f32) * step;

    let i1 = x_row + (num_vertices * y_row) + 1;
    let x1 = x0 + step;
    let y1 = y0;

    vertices[i0] = (x0, y0);
    vertices[i1] = (x1, y1);
  }

  for y_row in 1..num_vertices_normalized {
    for x_row in 0..num_vertices_normalized - 1 {
      let i0 = x_row + (num_vertices * (y_row - 1));
      let i1 = x_row + (num_vertices * (y_row - 1)) + 1;

      let (x0, y0) = vertices[i0];

      let i2 = x_row + (num_vertices * y_row);
      let x2 = x0;
      let y2 = y0 + step;

      let i3 = x_row + (num_vertices * y_row) + 1;
      let x3 = x0 + step;
      let y3 = y0 + step;

      vertices[i2] = (x2, y2);
      vertices[i3] = (x3, y3);

      indices.push(i2 as u32);
      indices.push(i1 as u32);
      indices.push(i0 as u32);

      indices.push(i3 as u32);
      indices.push(i1 as u32);
      indices.push(i2 as u32);
    }
  }

  let v: Vec<Vertex> = vertices.iter().map(|(x, y)| Vertex {
    position: [*x, 0.0f32, *y],
    color: [1.0, 1.0, 1.0],
    uv: [*x + size * 0.5, *y + size * 0.5],
  }).collect();

  return (v, indices);
}

#[test]
fn test_generate_plane() {
  generate_plane(5.0, 128);
}
