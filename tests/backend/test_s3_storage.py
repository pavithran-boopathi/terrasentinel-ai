import io

import boto3
import pytest
from moto import mock_aws

from backend.services.storage import S3Storage, StorageError


@pytest.fixture
def s3(tmp_path):
    with mock_aws():
        client = boto3.client("s3", region_name="eu-central-1")
        client.create_bucket(
            Bucket="ts-data", CreateBucketConfiguration={"LocationConstraint": "eu-central-1"}
        )
        yield S3Storage("ts-data", tmp_path / "cache", client=client), client, tmp_path


def test_save_stream_uploads_and_hashes(s3):
    st, client, _ = s3
    uri, sha, n = st.save_stream("images/a/scene.tif", io.BytesIO(b"hello"), 100)
    assert uri == "s3://ts-data/images/a/scene.tif" and n == 5 and len(sha) == 64
    assert client.get_object(Bucket="ts-data", Key="images/a/scene.tif")["Body"].read() == b"hello"
    with pytest.raises(StorageError):
        st.save_stream("images/b/x.tif", io.BytesIO(b"x" * 10), 5)


def test_workdir_commit_list_and_fresh_cache_download(s3, tmp_path):
    st, client, _ = s3
    d = st.workdir("outputs/j1")
    (d / "flood_mask.tif").write_bytes(b"mask")
    (d / "summary.json").write_text("{}")
    st.commit_dir("outputs/j1")
    assert st.list_names("s3://ts-data/outputs/j1") == ["flood_mask.tif", "summary.json"]
    fresh = S3Storage("ts-data", tmp_path / "other-cache", client=client)  # e.g. the API container
    assert fresh.local_path("s3://ts-data/outputs/j1/flood_mask.tif").read_bytes() == b"mask"
    st.delete_prefix("outputs/j1")
    assert st.list_names("s3://ts-data/outputs/j1") == []


def test_uri_validation(s3):
    st, _, _ = s3
    with pytest.raises(StorageError):
        st.local_path("s3://other-bucket/x")
    with pytest.raises(StorageError):
        st.local_path("s3://ts-data/../../etc/passwd")
