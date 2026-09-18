function run_preprocess_ecg_file(input_mat_path, output_mat_path)

    this_dir = fileparts(mfilename('fullpath'));
    addpath(this_dir);

    data = load(input_mat_path);

    if ~isfield(data, 'VGH')
        error('Input file does not contain VGH.');
    end

    if ~isfield(data.VGH, 'ecg')
        error('VGH does not contain an ecg channel.');
    end

    ecg = double(data.VGH.ecg.signal(:));
    Fs = double(data.VGH.ecg.fs);

    [prep_ecg, Q, R, S, B] = preprocess_ecg(ecg, Fs);

    fs = 1000;

    save(output_mat_path, ...
        'prep_ecg', ...
        'Q', ...
        'R', ...
        'S', ...
        'B', ...
        'fs', ...
        '-v7');
end
