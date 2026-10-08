from train_doctor import accel

NVSMI = "0, NVIDIA A100-SXM4-40GB, 37, 1234, 40960, 87.50\n1, NVIDIA A100-SXM4-40GB, 0, 3, 40960, [N/A]\n"


def test_parse_nvidia_smi():
    gpus = accel.parse_nvidia_smi(NVSMI)
    assert len(gpus) == 2
    assert gpus[0]["name"] == "NVIDIA A100-SXM4-40GB"
    assert gpus[0]["util_pct"] == 37.0 and gpus[0]["mem_total_mib"] == 40960.0
    assert gpus[1]["power_w"] is None


def test_fake_nvidia_smi_on_path(fake_bin):
    fake_bin("nvidia-smi", f"printf '{NVSMI}'\n")
    assert accel.gpu_backend() == "nvidia"
    sample = accel.sample_gpu("nvidia")
    assert sample["backend"] == "nvidia" and sample["gpus"][0]["util_pct"] == 37.0


def test_parse_ioreg():
    text = (
        '"PerformanceStatistics" = {"Tiler Utilization %"=10,"Device Utilization %"=52,"Renderer Utilization %"=51}\n"model" = "Apple M4"'
    )
    out = accel.parse_ioreg(text)
    assert out == {"util_pct": 52.0, "name": "Apple M4"}
    assert accel.parse_ioreg("nothing here") is None
