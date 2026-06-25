import random

# Adapted from Aero25x/random-user-agents.
# The original repository is MIT licensed:
# https://github.com/Aero25x/random-user-agents

DESKTOP_TARGETS = (
    ("windows", "chrome"),
    ("windows", "firefox"),
    ("ubuntu", "chrome"),
    ("ubuntu", "firefox"),
)

CHROME_MAJOR_VERSIONS = range(125, 139)
FIREFOX_MAJOR_VERSIONS = range(120, 136)


def generate_random_user_agent(device_type=None, browser_type=None):
    if device_type is None or browser_type is None:
        device_type, browser_type = random.choice(DESKTOP_TARGETS)

    if device_type == "windows":
        platform = "Windows NT 10.0; Win64; x64"
    elif device_type == "ubuntu":
        platform = "X11; Ubuntu; Linux x86_64"
    else:
        raise ValueError(f"Unsupported desktop device type: {device_type}")

    if browser_type == "chrome":
        major_version = random.choice(CHROME_MAJOR_VERSIONS)
        return (
            f"Mozilla/5.0 ({platform}) AppleWebKit/537.36 (KHTML, like Gecko) "
            f"Chrome/{major_version}.0.0.0 Safari/537.36"
        )

    if browser_type == "firefox":
        major_version = random.choice(FIREFOX_MAJOR_VERSIONS)
        return (
            f"Mozilla/5.0 ({platform}; rv:{major_version}.0) "
            f"Gecko/20100101 Firefox/{major_version}.0"
        )

    raise ValueError(f"Unsupported desktop browser type: {browser_type}")
