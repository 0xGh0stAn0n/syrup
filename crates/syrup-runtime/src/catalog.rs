//! Targets Syrup can find and the capability that detects each one. Adding a
//! target is a new entry here; the grammar, planner and generator are shared.

use serde::{Deserialize, Serialize};

use crate::abi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Face,
    Word,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    FaceDetection,
    TextRecognition,
}

impl Capability {
    pub fn abi_id(self) -> u32 {
        match self {
            Capability::FaceDetection => abi::SYRUP_CAP_FACE,
            Capability::TextRecognition => abi::SYRUP_CAP_TEXT,
        }
    }

    pub fn from_abi_id(id: u32) -> Option<Self> {
        match id {
            abi::SYRUP_CAP_FACE => Some(Capability::FaceDetection),
            abi::SYRUP_CAP_TEXT => Some(Capability::TextRecognition),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Capability::FaceDetection => "face_detection",
            Capability::TextRecognition => "text_recognition",
        }
    }
}

#[derive(Debug)]
pub struct TargetEntry {
    pub target: Target,
    pub label: &'static str,
    // Noun phrases, words joined by `_`.
    pub singular: &'static [&'static str],
    pub plural: &'static [&'static str],
    pub capability: Capability,
    pub default_min_confidence: f32,
    pub keypoints: &'static [&'static str],
    pub description: &'static str,
}

pub const TARGETS: &[TargetEntry] = &[
    TargetEntry {
        target: Target::Face,
        label: "face",
        singular: &["face", "human_face"],
        plural: &["faces", "human_faces"],
        capability: Capability::FaceDetection,
        // YuNet stays below ~0.4 on the non-face fixtures and above ~0.85 on faces.
        default_min_confidence: 0.6,
        // YuNet's order; right/left are the subject's.
        keypoints: &[
            "right_eye",
            "left_eye",
            "nose_tip",
            "right_mouth_corner",
            "left_mouth_corner",
        ],
        description: "where human faces are; never who, or how they look",
    },
    TargetEntry {
        target: Target::Word,
        label: "word",
        singular: &["word"],
        plural: &["words"],
        capability: Capability::TextRecognition,
        // Tesseract scores legible print above 0.9; below half it is guessing.
        default_min_confidence: 0.5,
        keypoints: &[],
        description: "words of printed text, with what they say",
    },
];

pub fn entry(target: Target) -> &'static TargetEntry {
    TARGETS
        .iter()
        .find(|e| e.target == target)
        .expect("every target is in the catalog")
}

pub fn entry_for_capability(capability: Capability) -> &'static TargetEntry {
    TARGETS
        .iter()
        .find(|e| e.capability == capability)
        .expect("every capability is in the catalog")
}

/// A target by any of its nouns, e.g. `face` or `human_faces`.
pub fn target_named(noun: &str) -> Option<Target> {
    TARGETS
        .iter()
        .find(|e| e.singular.contains(&noun) || e.plural.contains(&noun))
        .map(|e| e.target)
}

pub fn known_targets() -> String {
    TARGETS
        .iter()
        .map(|e| format!("{} ({})", e.plural[0], e.description))
        .collect::<Vec<_>>()
        .join("; ")
}
