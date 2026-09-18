import numpy as np
from scipy.io import loadmat


def load_apnea_labels(
    mat_path,
    label_name="apnea_1a",
):
    """
    Load an apnea annotation channel from the VGH structure.
    """
    data = loadmat(
        mat_path,
        variable_names=["VGH"],
    )

    vgh = data["VGH"][0, 0]

    if label_name not in vgh.dtype.names:
        raise KeyError(
            f"{label_name!r} not found in VGH."
        )

    obj = vgh[label_name][0, 0]

    labels = np.asarray(
        obj["signal"]
    ).squeeze()

    fs = float(
        np.asarray(
            obj["fs"]
        ).squeeze()
    )

    return labels, fs


def resample_labels_nearest(
    labels,
    label_fs,
    n_target,
    target_fs,
):
    """
    Map categorical labels to a target sampling grid using
    nearest-neighbor time alignment.
    """
    labels = np.asarray(
        labels
    ).squeeze()

    target_times = (
        np.arange(n_target)
        / target_fs
    )

    source_indices = np.rint(
        target_times * label_fs
    ).astype(int)

    source_indices = np.clip(
        source_indices,
        0,
        len(labels) - 1,
    )

    return labels[
        source_indices
    ]


def block_mask_to_samples(
    block_mask,
    n_samples,
    fs,
    block_seconds=120.0,
):
    """
    Expand a block-level Boolean mask onto the sample grid.
    """
    block_mask = np.asarray(
        block_mask,
        dtype=bool,
    )

    samples_per_block = int(
        round(
            block_seconds * fs
        )
    )

    sample_mask = np.zeros(
        n_samples,
        dtype=bool,
    )

    for block_index, keep in enumerate(
        block_mask
    ):
        if not keep:
            continue

        start = (
            block_index
            * samples_per_block
        )

        stop = min(
            start + samples_per_block,
            n_samples,
        )

        sample_mask[
            start:stop
        ] = True

    return sample_mask


def build_training_masks(
    *,
    block_rqi,
    qc_pass_blocks,
    apnea_labels,
    n_samples,
    fs,
    rqi_threshold=0.5,
    block_seconds=120.0,
):
    """
    Construct respiratory training masks.

    Non-apnea:
        QC pass
        RQI >= threshold
        apnea == 0

    Apnea:
        QC pass
        apnea != 0

    The apnea training set is not screened by RQI.
    """
    block_rqi = np.asarray(
        block_rqi,
        dtype=float,
    )

    qc_pass_blocks = np.asarray(
        qc_pass_blocks,
        dtype=bool,
    )

    apnea_labels = np.asarray(
        apnea_labels
    ).squeeze()

    if len(apnea_labels) != n_samples:
        raise ValueError(
            "apnea_labels must be aligned to the target "
            "sample grid."
        )

    qc_mask = block_mask_to_samples(
        qc_pass_blocks,
        n_samples,
        fs,
        block_seconds=block_seconds,
    )

    rqi_pass_blocks = (
        np.isfinite(block_rqi)
        & (
            block_rqi
            >= float(rqi_threshold)
        )
    )

    rqi_mask = block_mask_to_samples(
        rqi_pass_blocks,
        n_samples,
        fs,
        block_seconds=block_seconds,
    )

    apnea_positive = (
        apnea_labels != 0
    )

    no_apnea = (
        apnea_labels == 0
    )

    non_apnea_mask = (
        qc_mask
        & rqi_mask
        & no_apnea
    )

    apnea_mask = (
        qc_mask
        & apnea_positive
    )

    return {
        "qc_mask": qc_mask,
        "rqi_mask": rqi_mask,
        "rqi_pass_blocks": rqi_pass_blocks,
        "apnea_positive": apnea_positive,
        "no_apnea": no_apnea,
        "non_apnea_mask": non_apnea_mask,
        "apnea_mask": apnea_mask,
    }


def prepare_apnea_screening(
    mat_path,
    *,
    block_rqi,
    qc_pass_blocks,
    n_samples,
    fs,
    rqi_threshold=0.5,
    apnea_label="apnea_1a",
    block_seconds=120.0,
):
    """
    Load apnea annotations, align them to the respiratory
    sampling grid, and construct the training masks.
    """
    apnea_raw, apnea_fs = (
        load_apnea_labels(
            mat_path,
            label_name=apnea_label,
        )
    )

    apnea = resample_labels_nearest(
        apnea_raw,
        apnea_fs,
        n_samples,
        fs,
    )

    masks = build_training_masks(
        block_rqi=block_rqi,
        qc_pass_blocks=qc_pass_blocks,
        apnea_labels=apnea,
        n_samples=n_samples,
        fs=fs,
        rqi_threshold=rqi_threshold,
        block_seconds=block_seconds,
    )

    return {
        "apnea": apnea,
        "apnea_fs": apnea_fs,
        **masks,
    }


def select_training_mask(
    screening,
    mode,
):
    """
    Select the respiratory training dataset.
    """
    if mode == "non_apnea":
        return screening[
            "non_apnea_mask"
        ]

    if mode == "apnea":
        return screening[
            "apnea_mask"
        ]

    raise ValueError(
        "mode must be 'non_apnea' or 'apnea'."
    )
