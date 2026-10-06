"""Provider failures. Messages are fixed so client responses cannot leak secrets."""


class ProviderUnavailable(Exception):
    """The provider could not be reached or rejected the call."""

    detail = "Interpretation provider is unavailable."


class ProviderBadResponse(Exception):
    """The provider replied with a payload that failed structured validation."""

    detail = "Interpretation provider returned an invalid response."
