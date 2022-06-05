pub fn compute_work_group_count(
  (width, height): (u32, u32),
  (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
  let x = (width + workgroup_width - 1) / workgroup_width;
  let y = (height + workgroup_height - 1) / workgroup_height;

  return (x, y);
}

#[inline]
pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
    debug_assert!(min <= max, "min must be less than or equal to max");
    if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    }
}

