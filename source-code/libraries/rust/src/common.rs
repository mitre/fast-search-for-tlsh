pub const MAX_HEADER_TRIANGLE_INEQUALITY_VIOLATION: i32 = 46;
pub const TLSH_DIGEST_LENGTH: usize = 70;

pub(crate) const fn mod_diff(x: i32, y: i32, n: i32) -> i32 {
    let delta = (x - y).abs();
    let n_minus_delta = (n - delta).abs();

    if delta < n_minus_delta {
        delta
    } else {
        n_minus_delta
    }
}

#[derive(Hash, Eq, Clone, Debug, PartialEq)]
pub struct TlshDigest {
    pub checksum: u8,
    pub l: u8,
    pub q1: u8,
    pub q2: u8,
    pub body: [u8; 32],
}

const fn calculate_body_byte_diff(x: u8, y: u8) -> i32 {
    let x_high = x >> 4;
    let x_low = x & 0x0F;
    let y_high = y >> 4;
    let y_low = y & 0x0F;

    let x1 = x_high >> 2;
    let x2 = x_high & 0b11;
    let x3 = x_low >> 2;
    let x4 = x_low & 0b11;

    let y1 = y_high >> 2;
    let y2 = y_high & 0b11;
    let y3 = y_low >> 2;
    let y4 = y_low & 0b11;

    let d1 = (x1 as i32 - y1 as i32).abs();
    let d2 = (x2 as i32 - y2 as i32).abs();
    let d3 = (x3 as i32 - y3 as i32).abs();
    let d4 = (x4 as i32 - y4 as i32).abs();

    (if d1 == 3 { 6 } else { d1 }
        + if d2 == 3 { 6 } else { d2 }
        + if d3 == 3 { 6 } else { d3 }
        + if d4 == 3 { 6 } else { d4 })
}

const fn generate_body_lookup_table() -> [[i32; 256]; 256] {
    let mut table = [[0; 256]; 256];
    let mut x = 0;
    while x < 256 {
        let mut y = 0;
        while y < 256 {
            table[x][y] = calculate_body_byte_diff(x as u8, y as u8);
            y += 1;
        }
        x += 1;
    }
    table
}

const fn generate_q_lookup_table() -> [[i32; 256]; 256] {
    let mut table = [[0; 256]; 256];
    let mut x: usize = 0;
    while x < 256 {
        let mut y: usize = 0;
        while y < 256 {
            let diff = mod_diff(x as i32, y as i32, 16);
            let result = if diff <= 1 { diff } else { (diff - 1) * 12 };
            table[x][y] = result;
            y += 1;
        }
        x += 1;
    }
    table
}

impl TlshDigest {
    pub(crate) const BODY_DIFF_LOOKUP_TABLE: [[i32; 256]; 256] = generate_body_lookup_table();
    pub(crate) const Q_DIFF_LOOKUP_TABLE: [[i32; 256]; 256] = generate_q_lookup_table();

    /// Loads the raw byte form of a digest, where hex digits have been converted into their
    /// corresponding char values.
    ///
    /// Not to be used for plaintext strings. For that, use from_plaintext.
    ///
    /// # Arguments
    ///
    /// * `hex`: A
    ///
    /// returns: TlshDigest
    pub fn from_bytes(bytes: &[u8; TLSH_DIGEST_LENGTH]) -> Self {
        let checksum = (bytes[0] << 4) | bytes[1];
        let l = bytes[3] << 4 | bytes[2];
        let q1 = bytes[4];
        let q2 = bytes[5];

        let mut body = [0u8; 32];

        for i in 0..32 {
            body[i] = (bytes[6 + 2 * i] << 4) | bytes[7 + 2 * i];
        }

        TlshDigest {
            checksum,
            l,
            q1,
            q2,
            body,
        }
    }

    /// # Arguments
    ///
    /// * `plaintext`: An old-version (non-"T1"-prefixed) plaintext TLSH digest.
    ///
    /// returns: TlshDigest
    pub fn from_plaintext(plaintext: &[u8; TLSH_DIGEST_LENGTH]) -> Self {
        const HEX_TO_BYTE_LOOKUP_TABLE: [u8; 256] = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        Self::from_bytes(&plaintext.map(|c| HEX_TO_BYTE_LOOKUP_TABLE[c as usize]))
    }

    pub fn to_hex(&self) -> [u8; TLSH_DIGEST_LENGTH] {
        let mut hex = [0u8; TLSH_DIGEST_LENGTH];
        hex[0] = self.checksum >> 4;
        hex[1] = self.checksum & 0x0F;
        hex[2] = self.l & 0b1111;
        hex[3] = self.l >> 4;
        hex[4] = self.q1;
        hex[5] = self.q2;

        for i in 0..32 {
            hex[6 + 2 * i] = self.body[i] >> 4;
            hex[7 + 2 * i] = self.body[i] & 0x0F;
        }

        hex
    }

    pub fn distance_headers(t_x: &TlshDigest, t_y: &TlshDigest) -> i32 {
        let mut diff = 0;

        if t_x.checksum != t_y.checksum {
            diff += 1;
        }

        let l_diff = mod_diff(t_x.l as i32, t_y.l as i32, 256);
        diff += match l_diff {
            0 => 0,
            1 => 1,
            _ => l_diff * 12,
        };

        // It works: if you're curious, you can go back to this slow-path, or test it as follows:
        // let q1_diff = mod_diff(t_x.q1 as i32, t_y.q1 as i32, 16);
        // diff += if q1_diff <= 1 {
        //     q1_diff
        // } else {
        //     (q1_diff - 1) * 12
        // };
        // assert_eq!(q1_diff, Self::Q_DIFF_LOOKUP_TABLE[t_x.q1 as usize][t_y.q1 as usize]);
        // Repeat for q2, which is computed in the same way, only with t_x.q2 and t_y.q2.

        diff += Self::Q_DIFF_LOOKUP_TABLE[t_x.q1 as usize][t_y.q1 as usize];
        diff += Self::Q_DIFF_LOOKUP_TABLE[t_x.q2 as usize][t_y.q2 as usize];

        diff
    }

    pub fn distance_bodies(t_x: &TlshDigest, t_y: &TlshDigest) -> i32 {
        let mut diff = 0;

        for i in 0..32 {
            let x = t_x.body[i];
            let y = t_y.body[i];
            diff += Self::BODY_DIFF_LOOKUP_TABLE[x as usize][y as usize];
        }

        diff
    }
}

// For plaintext TLSH digests, for those who want it.
pub fn plaintext_distance_headers(
    t_x: &[u8; TLSH_DIGEST_LENGTH],
    t_y: &[u8; TLSH_DIGEST_LENGTH],
) -> i32 {
    let mut diff = 0;

    // Checksum
    if t_x[0] != t_y[0] || t_x[1] != t_y[1] {
        diff += 1;
    }

    // Length
    let l_value_t_x = (t_x[3] as i32) * 16 + (t_x[2] as i32);
    let l_value_t_y = (t_y[3] as i32) * 16 + (t_y[2] as i32);
    let l_diff = mod_diff(l_value_t_x, l_value_t_y, 256);
    diff += match l_diff {
        0 => 0,
        1 => 1,
        _ => l_diff * 12,
    };

    // Q ratios
    let q_t_x = (t_x[4] as i32) * 16 + (t_x[5] as i32);
    let q_t_y = (t_y[4] as i32) * 16 + (t_y[5] as i32);
    let q1_t_x = q_t_x & 0x0F;
    let q1_t_y = q_t_y & 0x0F;
    let q2_t_x = (q_t_x & 0xF0) >> 4;
    let q2_t_y = (q_t_y & 0xF0) >> 4;

    let q1_diff = mod_diff(q1_t_x, q1_t_y, 16);
    diff += if q1_diff <= 1 {
        q1_diff
    } else {
        (q1_diff - 1) * 12
    };

    let q2_diff = mod_diff(q2_t_x, q2_t_y, 16);
    diff += if q2_diff <= 1 {
        q2_diff
    } else {
        (q2_diff - 1) * 12
    };

    diff
}

// For plaintext TLSH digests, for those who want it.
pub fn plaintext_distance_bodies(
    t_x: &[u8; TLSH_DIGEST_LENGTH],
    t_y: &[u8; TLSH_DIGEST_LENGTH],
) -> i32 {
    let mut diff = 0;

    // Index at which point the TLSH "body" starts
    const BODY_INDEX: usize = 6;

    for i in 0..64 {
        let x = t_x[BODY_INDEX + i];
        let y = t_y[BODY_INDEX + i];

        let x1 = x >> 2;
        let x2 = x & 0b11;
        let y1 = y >> 2;
        let y2 = y & 0b11;

        let d1 = (x1 as i32 - y1 as i32).abs();
        let d2 = (x2 as i32 - y2 as i32).abs();

        diff += if d1 == 3 { 6 } else { d1 };
        diff += if d2 == 3 { 6 } else { d2 };
    }

    diff
}
