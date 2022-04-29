import json

from metaflow import S3, FlowSpec, step


class S3DemoFlow(FlowSpec):
    @step
    def start(self):
        with S3(run=self) as s3:
            message = json.dumps({"message": "hello world"})
            url = s3.put("example_object", message)
            print("Message saved at", url)
        self.next(self.end)

    @step
    def end(self):
        with S3(run=self) as s3:
            s3obj = s3.get("example_object")
            print("Object found at", s3obj.url)
            print("Message", json.loads(s3obj.text))


if __name__ == "__main__":
    S3DemoFlow()
