use stime::Device;

#[test]
fn all_returns_one_device_per_counted_device() {
    // Given the system's device count
    let count = Device::count().unwrap();

    // When listing all devices
    let devices = Device::all().unwrap();

    // Then there is one device per counted device, in ordinal order
    assert_eq!(devices.len(), count);
    for (i, device) in devices.iter().enumerate() {
        assert_eq!(device.ordinal(), i);
    }
}

#[test]
fn get_with_out_of_range_ordinal_fails() {
    // Given an ordinal past the last device
    let ordinal = Device::count().unwrap();

    // When getting the device
    let result = Device::get(ordinal);

    // Then it returns an error
    assert!(result.is_err());
}

#[test]
fn device_reports_its_properties() {
    // Given the first device
    let device = Device::get(0).unwrap();

    // When querying its properties
    let name = device.name().unwrap();
    let total_memory = device.total_memory().unwrap();
    let (major, _minor) = device.compute_capability().unwrap();

    // Then they hold plausible values
    assert!(!name.is_empty());
    assert!(total_memory > 0);
    assert!(major >= 1);
}
