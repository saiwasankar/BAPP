"""Domain errors. main.py maps each one to a JSON error response."""


class BappsError(Exception):
    status_code = 400
    code = "BAD_REQUEST"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(BappsError):
    status_code = 404
    code = "NOT_FOUND"


class ConflictError(BappsError):
    status_code = 409
    code = "CONFLICT"


class ForbiddenError(BappsError):
    status_code = 403
    code = "FORBIDDEN"


class InvalidRequestError(BappsError):
    status_code = 422
    code = "INVALID_REQUEST"
