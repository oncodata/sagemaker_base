import shutil
import boto3
import os
import glob
import builtins
import slideio


class S3FileManager:
    def __init__(self, bucket_name, local_dir):
        # if the bucket_name has subfolders inside, separate it into prefix and bucket_name with only the real bucket name
        if '/' in bucket_name:
            bucket_name, prefix = bucket_name.split('/', 1)
        else:
            prefix = ''
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.local_dir = local_dir
        self.file_sizes_dict = {}
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Erase contents of local_dir, including directories and files
        for root, dirs, files in os.walk(local_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # Download the file structure from the bucket and replicate the same in local_dir, but with empty files only
        self.s3 = boto3.client('s3')
        s3_objects = self.s3.list_objects(Bucket=bucket_name, Prefix=prefix)
        continuation_token = None
        while True:
            params = {'Bucket': bucket_name, 'Prefix': prefix}
            if continuation_token:
                params['ContinuationToken'] = continuation_token
            s3_objects = self.s3.list_objects_v2(**params)
            if 'Contents' in s3_objects:
                for obj in s3_objects['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        # Skip directory-like objects
                        continue
                    # save the obj['Size'] in Gb
                    self.file_sizes_dict[key] = obj['Size']
                    local_file_path = os.path.join(local_dir, os.path.relpath(key, prefix))
                    local_file_dir = os.path.dirname(local_file_path)
                    if not os.path.exists(local_file_dir):
                        os.makedirs(local_file_dir)
                    if not os.path.exists(local_file_path):
                        with open(local_file_path, 'w'):
                            pass
            if not s3_objects.get('NextContinuationToken'):
                break
            continuation_token = s3_objects['NextContinuationToken']

        # Store a reference to the original open method
        original_open = builtins.open

        def open_with_hook(file, mode='r', **kwargs):
            # Custom code to execute before opening the file
            if os.path.isfile(file):
                self.use_file(file)
            # Call the original open method with the provided arguments
            return original_open(file, mode=mode, **kwargs)

        # Override the open method with our custom version
        if 'open_with_hook' not in str(builtins.open):
            builtins.open = open_with_hook
        else:
            print("Apenas uma instância de S3FileManager pode funcionar ao mesmo tempo, reinicie o kernel!")
            return

        def open_slide_with_handler(old_func, slide_src, driver):
            self.use_file(slide_src)
            return old_func(slide_src, driver)

        old_function = slideio.open_slide
        new_func_lambda = lambda img_path, driver='AUTO': open_slide_with_handler(old_function, img_path, driver)
        slideio.open_slide = new_func_lambda

    def use_file(self, file_path):
        # If the file_path points to a file that has size larger than 0, just return the file_path
        if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
            return file_path

        # Get the relative file path from the file_path by removing the local_dir from the file_path
        relative_file_path = os.path.relpath(file_path, self.local_dir)

        # Get the file key from s3 by joining self.prefix with the relative_file_path
        key = os.path.join(self.prefix, relative_file_path)

        # Get the available storage space
        available_storage = self.calculate_available_storage()
        # if there is less than 5 GB of storage available, delete the least recently accessed file that has size larger than 0
        if available_storage < 5 * 1024 * 1024 * 1024:
            available_storage = self.delete_unused_files(available_storage)
        # Download from S3 the file and replace the 0 sized file in file_path
        self.s3.download_file(self.bucket_name, key, file_path)

        return file_path

    def get_file_size(self, file_path):
        # Get the file key from s3 by joining self.prefix with the file_path
        relative_file_path = os.path.relpath(file_path, self.local_dir)
        key = os.path.join(self.prefix, relative_file_path)
        return self.file_sizes_dict.get(key, 0)

    # Function that calculates available disk space
    def calculate_available_storage(self):
        total_capacity = shutil.disk_usage(self.local_dir).total
        total_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(self.local_dir, '*')))
        return total_capacity - total_size

    def delete_unused_files(self, available_storage=None):
        # Delete non 0 sized files that were not accessed for recently until there is 5 GB of storage available
        files_to_delete = []
        # Make a list of all files that have more than 0 bytes
        for file in os.listdir(self.local_dir):
            file_path = os.path.join(self.local_dir, file)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                files_to_delete.append(file_path)
        # Sort the files from least recently accessed to most recently accessed
        files_to_delete.sort(key=os.path.getatime)
        if available_storage is None:
            available_storage = self.calculate_available_storage()
        # Delete the least recently accessed file until there is 5 GB of storage available
        # Add to available_storage the size of the deleted file in order to calculate what is left to delete
        for file in files_to_delete:
            if available_storage < 5 * 1024 * 1024 * 1024:
                available_storage += os.path.getsize(file)
                # Replace file with a 0 sized file
                with open(file, 'w'):
                    pass
                # print(f"Deleted file {file}")
            else:
                break
        return available_storage


class S3UploadSync:
    def __init__(self, bucket_name, local_dir, bucket_key):
        self.bucket_name = bucket_name
        self.local_dir = local_dir
        self.bucket_key = bucket_key
        self.s3 = boto3.client('s3')

    def sync(self):
        # Upload all files from local_dir to the bucket with the bucket_key
        uploaded_files_count = 0
        for root, dirs, files in os.walk(self.local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_file_path = os.path.relpath(file_path, self.local_dir)
                key = os.path.join(self.bucket_key, relative_file_path)
                self.s3.upload_file(file_path, self.bucket_name, key)
                uploaded_files_count += 1
        # Print the number of uploaded files
        print(f"Uploaded {uploaded_files_count} files to {self.bucket_name}")
