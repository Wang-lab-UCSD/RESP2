mod symbol_readers {


    fn standard_symbol_set_char_reader(char &letter) -> i32 {
        let mut i32 keycode = -1;
        match letter {
            'A' => keycode = 0,
            'C' => keycode = 1,
            'D' => keycode = 2,
            'E' => keycode = 3,
            'F' => keycode = 4,
            'G' => keycode = 5,
            'H' => keycode = 6,
            'I' => keycode = 7,
            'K' => keycode = 8,
            'L' => keycode = 9,
            'M' => keycode = 10,
            'N' => keycode = 11,
            'P' => keycode = 12,
            'Q' => keycode = 13,
            'R' => keycode = 14,
            'S' => keycode = 15,
            'T' => keycode = 16,
            'V' => keycode = 17,
            'W' => keycode = 18,
            'Y' => keycode = 19,
            _ => return keycode;
        }
        return keycode;
    }


    fn gapped_symbol_set_char_reader(char &letter) -> i32 {
        let mut i32 keycode = -1;
        match letter {
            'A' => keycode = 0,
            'C' => keycode = 1,
            'D' => keycode = 2,
            'E' => keycode = 3,
            'F' => keycode = 4,
            'G' => keycode = 5,
            'H' => keycode = 6,
            'I' => keycode = 7,
            'K' => keycode = 8,
            'L' => keycode = 9,
            'M' => keycode = 10,
            'N' => keycode = 11,
            'P' => keycode = 12,
            'Q' => keycode = 13,
            'R' => keycode = 14,
            'S' => keycode = 15,
            'T' => keycode = 16,
            'V' => keycode = 17,
            'W' => keycode = 18,
            'Y' => keycode = 19,
            '-' => keycode = 20,
            _ => return keycode;
        }
        return keycode;
    }


    fn expanded_symbol_set_char_reader(char &letter) -> i32 {
        let mut i32 keycode = -1;
        match letter {
            'A' => keycode = 0,
            'C' => keycode = 1,
            'D' => keycode = 2,
            'E' => keycode = 3,
            'F' => keycode = 4,
            'G' => keycode = 5,
            'H' => keycode = 6,
            'I' => keycode = 7,
            'K' => keycode = 8,
            'L' => keycode = 9,
            'M' => keycode = 10,
            'N' => keycode = 11,
            'P' => keycode = 12,
            'Q' => keycode = 13,
            'R' => keycode = 14,
            'S' => keycode = 15,
            'T' => keycode = 16,
            'V' => keycode = 17,
            'W' => keycode = 18,
            'Y' => keycode = 19,
            '-' => keycode = 20,
            'B' => keycode = 21,
            'J' => keycode = 22,
            'O' => keycode = 23,
            'U' => keycode = 24,
            'X' => keycode = 25,
            'Z' => keycode = 26,
            _ => return keycode;
        }
        return keycode;
    }

}
