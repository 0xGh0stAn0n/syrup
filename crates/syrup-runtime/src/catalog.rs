//! Targets Syrup can find and how each one is found. Adding a target is a
//! new entry here; the grammar, planner and generator are shared.

use serde::{Deserialize, Serialize};

use crate::abi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Face,
    Word,
    Region,
    Bar,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Cyan,
    Blue,
    Purple,
    Magenta,
}

#[derive(Debug)]
pub struct ColorEntry {
    pub color: Color,
    pub names: &'static [&'static str],
    /// Hue range in degrees, wrapping through 360 when the start is larger.
    pub hue: (u32, u32),
}

/// Neighbouring hue ranges, so every saturated colour has exactly one name.
pub const COLORS: &[ColorEntry] = &[
    ColorEntry {
        color: Color::Red,
        names: &["red"],
        hue: (340, 20),
    },
    ColorEntry {
        color: Color::Orange,
        names: &["orange"],
        hue: (20, 45),
    },
    ColorEntry {
        color: Color::Yellow,
        names: &["yellow"],
        hue: (45, 70),
    },
    ColorEntry {
        color: Color::Green,
        names: &["green"],
        hue: (70, 165),
    },
    ColorEntry {
        color: Color::Cyan,
        names: &["cyan"],
        hue: (165, 195),
    },
    ColorEntry {
        color: Color::Blue,
        names: &["blue"],
        hue: (195, 255),
    },
    ColorEntry {
        color: Color::Purple,
        names: &["purple", "violet"],
        hue: (255, 290),
    },
    ColorEntry {
        color: Color::Magenta,
        names: &["magenta"],
        hue: (290, 340),
    },
];

// Below these a pixel is grey, white or black rather than a colour; the
// core's bar detection uses the same thresholds.
pub const MIN_SATURATION_PCT: u32 = 35;
pub const MIN_VALUE_PCT: u32 = 30;

impl Color {
    pub fn name(self) -> &'static str {
        color(self).names[0]
    }
}

pub fn color(color: Color) -> &'static ColorEntry {
    COLORS
        .iter()
        .find(|c| c.color == color)
        .expect("every colour is in the table")
}

/// How a target is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finder {
    /// By a detector the host provides.
    Detect(Capability),
    /// By the generated module: runs of one colour, grouped into regions.
    Color {
        min_run: u32,
        min_height: u32,
        max_gap: u32,
        /// Keep regions at least this many times wider than tall.
        min_aspect: Option<u32>,
    },
}

#[derive(Debug)]
pub struct TargetEntry {
    pub target: Target,
    pub label: &'static str,
    // Noun phrases, words joined by `_`.
    pub singular: &'static [&'static str],
    pub plural: &'static [&'static str],
    pub finder: Finder,
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
        finder: Finder::Detect(Capability::FaceDetection),
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
        finder: Finder::Detect(Capability::TextRecognition),
        // Tesseract scores legible print above 0.9; below half it is guessing.
        default_min_confidence: 0.5,
        keypoints: &[],
        description: "words of printed text, with what they say",
    },
    TargetEntry {
        target: Target::Region,
        label: "region",
        singular: &["region", "blob"],
        plural: &["regions", "blobs"],
        finder: Finder::Color {
            min_run: 3,
            min_height: 3,
            max_gap: 1,
            min_aspect: None,
        },
        // Confidence is the share of the box's pixels that have the colour.
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "areas of one colour, e.g. red_regions",
    },
    TargetEntry {
        target: Target::Bar,
        label: "bar",
        singular: &["bar"],
        plural: &["bars"],
        // The core's find_color_bar uses runs of at least 8 and 2-row gaps.
        finder: Finder::Color {
            min_run: 8,
            min_height: 2,
            max_gap: 2,
            min_aspect: Some(3),
        },
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "bars of one colour, at least 3 times wider than tall, e.g. red_bars",
    },
];

pub fn entry(target: Target) -> &'static TargetEntry {
    TARGETS
        .iter()
        .find(|e| e.target == target)
        .expect("every target is in the catalog")
}

/// A target by any of its nouns, e.g. `face` or `human_faces`.
pub fn target_named(noun: &str) -> Option<Target> {
    TARGETS
        .iter()
        .find(|e| e.singular.contains(&noun) || e.plural.contains(&noun))
        .map(|e| e.target)
}

pub fn color_named(word: &str) -> Option<Color> {
    COLORS
        .iter()
        .find(|c| c.names.contains(&word))
        .map(|c| c.color)
}

pub fn known_targets() -> String {
    TARGETS
        .iter()
        .map(|e| format!("{} ({})", e.plural[0], e.description))
        .collect::<Vec<_>>()
        .join("; ")
}
