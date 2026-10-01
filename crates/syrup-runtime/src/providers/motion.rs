//! Moving regions, from the core's frame differencing. Each session owns one
//! of these, since it compares a frame with the one before it.

use std::sync::Mutex;

use image::RgbaImage;
use syrup::motion::{MotionConfig, extract_blobs, motion_mask};

use super::{Detections, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::catalog::Capability;
use crate::contract::ProviderInfo;
use crate::error::Result;

pub struct Motion {
    config: MotionConfig,
    previous: Mutex<Option<RgbaImage>>,
}

impl Motion {
    pub fn new(config: MotionConfig) -> Motion {
        Motion {
            config,
            previous: Mutex::new(None),
        }
    }

    /// Forgets the previous frame, e.g. when the searched region moves.
    pub fn reset(&self) {
        *self.previous.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

impl Provider for Motion {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::Motion.as_str(),
            name: "frame_difference",
            model_sha256: None,
            runtime: "rust",
        }
    }

    /// Nothing moves in the first frame, or in a frame whose size differs
    /// from the one before.
    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let current = view.to_rgba();
        let mut previous = self.previous.lock().unwrap_or_else(|e| e.into_inner());
        let mask = previous
            .as_ref()
            .and_then(|before| motion_mask(before, &current, self.config.diff_threshold));
        *previous = Some(current);
        let Some(mask) = mask else {
            return Ok(Detections::default());
        };
        let boxes = extract_blobs(&mask, &self.config)
            .into_iter()
            .map(|r| {
                let mut changed = 0u32;
                for y in r.y..r.y + r.h {
                    for x in r.x..r.x + r.w {
                        changed += (mask.get_pixel(x, y)[0] > 0) as u32;
                    }
                }
                SyrupDetection {
                    x: r.x as f32,
                    y: r.y as f32,
                    w: r.w as f32,
                    h: r.h as f32,
                    score: changed as f32 / r.area() as f32,
                    value: 0.0,
                    n_keypoints: 0,
                    keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
                    payload: 0,
                }
            })
            .collect();
        Ok(Detections {
            boxes,
            texts: vec![],
        })
    }
}
