import shutil
import s3fs
import os


class LocalOrS3Client:
    def __init__(self, init_s3: bool = False, **s3fs_kwargs):
        self.init_s3 = init_s3
        if init_s3:
            self.fs = s3fs.S3FileSystem(**s3fs_kwargs)

    def exists(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.exists(path, **kwargs)
        else:
            return os.path.exists(path)

    def ls(self, path, **kwargs):
        if path.startswith("s3://"):
            return self.fs.ls(path, **kwargs)
        else:
            return os.listdir(path)

    def get(self, rpath, lpath, **kwargs):
        """Get file(s) from remote to local. Supports both S3 and local paths."""
        if rpath.startswith("s3://"):
            assert lpath and not lpath.startswith("s3://"), "lpath cannot be a S3 path."
            return self.fs.get(rpath, lpath, **kwargs)
        else:
            # Local to local copy
            if os.path.isdir(rpath):
                os.makedirs(lpath, exist_ok=True)
                for item in os.listdir(rpath):
                    src = os.path.join(rpath, item)
                    dst = os.path.join(lpath, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            else:
                os.makedirs(os.path.dirname(lpath), exist_ok=True)
                shutil.copy2(rpath, lpath)

    def put(self, lpath, rpath, **kwargs):
        """Put file(s) from local to remote. Supports both S3 and local paths."""
        if rpath.startswith("s3://"):
            assert lpath and not lpath.startswith("s3://"), "lpath cannot be a S3 path."
            return self.fs.put(lpath, rpath, **kwargs)
        else:
            # Local to local copy
            os.makedirs(
                os.path.dirname(rpath) if os.path.dirname(rpath) else ".", exist_ok=True
            )
            if os.path.isdir(lpath):
                os.makedirs(rpath, exist_ok=True)
                for item in os.listdir(lpath):
                    src = os.path.join(lpath, item)
                    dst = os.path.join(rpath, item)
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
            else:
                shutil.copy2(lpath, rpath)

    def copy(self, src, dst, **kwargs):
        """Copy a file or directory from src to dst, replacing dst's contents.

        Supports all combinations of local and S3 paths. If dst already
        exists it is removed first, so dst ends up an exact copy of src
        rather than a merge of the two.
        """
        src_is_s3 = src.startswith("s3://")
        dst_is_s3 = dst.startswith("s3://")

        if not src_is_s3 and not dst_is_s3:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.copy2(src, dst)
        elif src_is_s3 and dst_is_s3:
            if self.fs.exists(dst):
                self.fs.rm(dst, recursive=True)
            self.fs.copy(src, dst, recursive=True, **kwargs)
        elif src_is_s3 and not dst_is_s3:
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            self.fs.get(src, dst, recursive=True, **kwargs)
        else:  # local src, S3 dst
            if self.fs.exists(dst):
                self.fs.rm(dst, recursive=True)
            self.fs.put(src, dst, recursive=True, **kwargs)

    _MODEL_CHECKPOINT_FILENAMES = frozenset({
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
    })
    _MODEL_CHECKPOINT_PREFIXES = ("model-", "pytorch_model-")  # sharded checkpoints

    def copy_model_checkpoint(self, src_dir, dst_dir):
        """Copy only the HF model/tokenizer checkpoint files from src_dir into
        dst_dir, leaving everything else in dst_dir untouched.

        Round directories double as both a model checkpoint location and the
        working directory for that round's own generation/likelihood/AR-sampling
        outputs, many of which use round-invariant filenames with "skip if
        already exists" checkpointing. A wholesale directory copy (``copy()``)
        would therefore also drag in the *previous* round's own pipeline
        outputs, silently short-circuiting the current round's fresh writes.
        This copies only the checkpoint files needed to load the model.
        """
        for name in self.ls(src_dir):
            basename = os.path.basename(name)
            is_checkpoint_file = basename in self._MODEL_CHECKPOINT_FILENAMES or (
                basename.startswith(self._MODEL_CHECKPOINT_PREFIXES)
                and (basename.endswith(".safetensors") or basename.endswith(".bin"))
            )
            if is_checkpoint_file:
                self.copy(
                    os.path.join(src_dir, basename), os.path.join(dst_dir, basename)
                )

    def open(self, fp, mode="rb", **kwargs):
        if fp.startswith("s3://"):
            return self.fs.open(fp, mode=mode, **kwargs)
        else:
            return open(fp, mode=mode, **kwargs)
