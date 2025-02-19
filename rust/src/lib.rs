use pyo3::{
    pymodule,
    types::{PyDict, PyModule, PyList},
    Bound, FromPyObject, PyObject, PyResult, Python,
};
use numpy::ndarray::{ArrayViewMut3, ArrayViewMut2};
use symbol_readers::{standard_symbol_set_char_reader,
            gapped_symbol_set_char_reader,
            expanded_symbol_set_char_reader};


#[pyfunction]
fn onehot_flat_encode_list(seqlist: Vec<String>,
        mut out_arr: ArrayViewMut2<'_, u8>,
        expanded_symbol_set: bool,
        use_gaps: bool) -> PyResult<String> {
    if out_arr.len_of(0) != seqlist.len() {
        return "Incorrect array size.";
    }

    if expanded_symbol_set {
        for (i, &seq) in seqlist.iter().enumerate() {
            if (seq.len() * 27) > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = expanded_symbol_set_char_reader(&letter);
                out_arr[[i,j*27 + keycode]] = 1;
            }
        }
    }
    else if use_gaps {
        for (i, &seq) in seqlist.iter().enumerate() {
            if (seq.len() * 21) > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = gapped_symbol_set_char_reader(&letter);
                out_arr[[i,j*21 + keycode]] = 1;
            }
        }
    }
    else {
        for (i, &seq) in seqlist.iter().enumerate() {
            if (seq.len() * 20) > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = standard_symbol_set_char_reader(&letter);
                out_arr[[i,j*20 + keycode]] = 1;
            }
        }
    }

    return "";
}

#[pyfunction]
fn onehot_3d_encode_list(seqlist: Vec<String>,
        mut out_arr: ArrayViewMut3<'_, u8>,
        expanded_symbol_set: bool,
        use_gaps: bool) -> PyResult<String> {
    if out_arr.len_of(0) != seqlist.len() {
        return "Incorrect array size.";
    }

    if expanded_symbol_set {
        if out_arr.len_of(2) != 27 {
            return "Incorrect array size.";
        }
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = expanded_symbol_set_char_reader(&letter);
                out_arr[[i,j,keycode]] = 1;
            }
        }
    }
    else if use_gaps {
        if out_arr.len_of(2) != 21 {
            return "Incorrect array size.";
        }
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = gapped_symbol_set_char_reader(&letter);
                out_arr[[i,j,keycode]] = 1;
            }
        }
    }
    else {
        if out_arr.len_of(2) != 20 {
            return "Incorrect array size.";
        }
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = standard_symbol_set_char_reader(&letter);
                out_arr[[i,j,keycode]] = 1;
            }
        }
    }

    return "";
}


#[pyfunction]
fn integer_encode_list(seqlist: Vec<String>,
        mut out_arr: ArrayViewMut2<'_, u8>,
        expanded_symbol_set: bool,
        use_gaps: bool) -> PyResult<String> {
    if out_arr.len_of(0) != seqlist.len() {
        return "Incorrect array size.";
    }

    if expanded_symbol_set {
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = expanded_symbol_set_char_reader(&letter);
                out_arr[[i,j]] = keycode;
            }
        }
    }
    else if use_gaps {
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = gapped_symbol_set_char_reader(&letter);
                out_arr[[i,j]] = keycode;
            }
        }
    }
    else {
        for (i, &seq) in seqlist.iter().enumerate() {
            if seq.len() > out_arr.len_of(1) {
                return "Incorrect array size."
            }
            for (j, &letter) in seq.chars().enumerate() {
                let keycode = standard_symbol_set_char_reader(&letter);
                out_arr[[i,j]] = keycode;
            }
        }
    }

    return "";
}

#[pymodule]
fn resp_toolkit_rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(onehot_flat_encode_list, m)?)?;
    m.add_function(wrap_pyfunction!(onehot_3d_encode_list, m)?)?;
    m.add_function(wrap_pyfunction!(integer_encode_list, m)?)?;
}
