"""Input device listing that shows the host API, so the device actually opened is explicit.

Windows lists the same microphone under several host APIs (MME, DirectSound, WASAPI, WDM-KS).
No device is chosen automatically and the system default is not assumed to be the built-in mic.
"""
SAMPLERATE = 16000
HOSTAPI_ORDER = {'MME': 0, 'Windows DirectSound': 1, 'Windows WASAPI': 2, 'Windows WDM-KS': 3}


def list_input_devices(sd):
    apis = sd.query_hostapis()
    try:
        default_input = int(sd.default.device[0])
    except Exception:
        default_input = -1
    items = []
    for index, d in enumerate(sd.query_devices()):
        if d['max_input_channels'] <= 0:
            continue
        api = apis[d['hostapi']]['name']
        mark = ' · Windows 기본 입력' if index == default_input else ''
        items.append(dict(index=index, name=d['name'], hostapi=api,
                          default_samplerate=float(d['default_samplerate']),
                          is_system_default=index == default_input,
                          label=f"{index}: {d['name']} [{api}]{mark}"))
    items.sort(key=lambda r: (HOSTAPI_ORDER.get(r['hostapi'], 9), r['index']))
    return items


def check_device(sd, item):
    """Raise with a readable message if the device cannot open mono float32 at 16 kHz."""
    try:
        sd.check_input_settings(device=item['index'], channels=1, dtype='float32', samplerate=SAMPLERATE)
    except Exception as exc:
        raise RuntimeError(f"{item['label']}\n이 입력은 16000 Hz 모노로 열 수 없습니다: {exc}\n"
                           "같은 마이크의 [MME] 항목을 선택해 보세요.") from exc


def log_fields(item, stream_info=None):
    fields = dict(device=item['label'], device_index=item['index'], device_name=item['name'],
                  device_hostapi=item['hostapi'], device_default_samplerate=item['default_samplerate'],
                  device_is_system_default=int(item['is_system_default']))
    if stream_info:
        fields.update({'stream_' + k: v for k, v in stream_info.items()})
    return fields
