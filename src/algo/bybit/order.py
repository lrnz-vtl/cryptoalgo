import json
import logging
from dataclasses import dataclass
from typing import Protocol, Any

import requests
import time
import hashlib
import hmac
import uuid

from requests import Session, Response

api_key = '1z56Qq7Rl6APC0LVls'
secret_key = 'zD3LJ5FaYNhTvXQmYRjK7p4EB8p5XkCg0IHB'


class RequestType(Protocol):
    @staticmethod
    def make_payload_str(payload: dict) -> str:
        ...

    @staticmethod
    def request_response(url: str, endPoint: str, httpClient: Session, headers: dict, payload_str: str) -> Response:
        ...


class GET(RequestType):
    @staticmethod
    def make_payload_str(payload: dict) -> str:
        return '&'.join([f'{k}={v}' for k, v in payload.items()])

    @staticmethod
    def request_response(url: str, endPoint: str, httpClient: Session, headers: dict, payload_str: str) -> Response:
        full_url = url + endPoint + "?" + payload_str
        return httpClient.request("GET", full_url, headers=headers)


class POST(RequestType):
    @staticmethod
    def make_payload_str(payload: dict) -> str:
        return json.dumps(payload)

    @staticmethod
    def request_response(url: str, endPoint: str, httpClient: Session, headers: dict, payload_str: str) -> Response:
        full_url = url + endPoint
        return httpClient.request("POST", full_url, headers=headers, data=payload_str)


class Connection:
    def __init__(self, api_key: str, secret_key: str):
        self.httpClient = requests.Session()
        self.recv_window = str(5000)
        self.url = "https://api-testnet.bybit.com"
        self.api_key = api_key
        self.secret_key = secret_key

        self.logger = logging.getLogger(__name__)

    def HTTP_Request(self, endPoint: str, request_type: RequestType, payload: dict):
        timestamp = str(int(time.time() * 10 ** 3))

        payload_str = request_type.make_payload_str(payload)

        signature = self.genSignature(timestamp, payload_str)
        self.logger.debug(f'{signature=}')

        headers = {
            'X-BAPI-API-KEY': api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': self.recv_window,
            'Content-Type': 'application/json'
        }
        self.logger.debug(f'{payload=}')

        response = request_type.request_response(self.url, endPoint, self.httpClient, headers, payload_str)

        # self.logger.debug(f'{full_url=}')
        self.logger.debug(f'{response.status_code=}')
        self.logger.debug(response.text)

        return response

    def genSignature(self, timestamp, payload):
        param_str = str(timestamp) + api_key + self.recv_window + payload
        hash = hmac.new(bytes(secret_key, "utf-8"), param_str.encode("utf-8"), hashlib.sha256)
        signature = hash.hexdigest()
        return signature


def get_position_info(conn: Connection, symbol: str):
    endpoint = "/contract/v3/private/position/list"
    method = GET
    params = {"symbol": symbol}
    response = conn.HTTP_Request(endpoint, method, params)
    assert response.ok
    return response.json()['result']


def get_ob(conn: Connection, symbol: str) -> Any:
    endpoint = "/derivatives/v3/public/order-book/L2"
    method = GET
    params = {"symbol": symbol}
    response = conn.HTTP_Request(endpoint, method, params)
    assert response.ok
    return response.json()['result']


@dataclass
class OrderResponse:
    order_link_id: Any
    response: dict


def place_order(conn: Connection, symbol: str, qty: float, side: str) -> OrderResponse:
    endpoint = "/contract/v3/private/order/create"
    method = POST
    orderLinkId = uuid.uuid4().hex

    assert side in ["Buy", "Sell"]

    params = {
        # "category": "linear",
        "symbol": symbol,
        "side": side,
        "positionIdx": 0,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GoodTillCancel",
        "orderLinkId": orderLinkId}

    response = conn.HTTP_Request(endpoint, method, params)
    assert response.ok
    return OrderResponse(
        order_link_id=orderLinkId,
        response=response.json()
    )


if __name__ == '__main__':
    conn = Connection(api_key=api_key, secret_key=secret_key)

    symbol = 'BTCUSDT'
    pos_info = get_position_info(conn, symbol)
    print(f'{pos_info=}')

    ob_info = get_ob(conn, symbol)
    print(f'{ob_info=}')

    order_info = place_order(conn, symbol, 0.010, "Sell")
    print(f'{order_info=}')
