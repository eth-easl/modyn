import boto3

from .base import BaseAdapter


class S3Adapter(BaseAdapter):
    """S3Adapter is a wrapper around the boto3 module, which provides a
    dictionary-like interface to an S3 bucket."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.__s3 = boto3.resource('s3')
        self.__bucket = self.__s3.Bucket(
            self._config['storage']['s3']['bucket'])

    def get(self, keys: list[str]) -> list[bytes]:

        data = []
        for key in keys:
            data.append(self.__bucket.Object(key).get()['Body'].read())
        return data

    def put(self, key: list[str], data: list[bytes]) -> None:
        for i in range(len(key)):
            self.__bucket.put_object(Key=key[i], Body=data[i])
