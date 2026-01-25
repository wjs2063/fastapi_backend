class ExternalAPIError(Exception):
    """외부 API 호출 시 발생하는 기본 예외"""

    def __init__(self, service_name: str, status_code: int, detail: str):
        self.service_name = service_name
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"[{service_name}] {status_code}: {detail}")
