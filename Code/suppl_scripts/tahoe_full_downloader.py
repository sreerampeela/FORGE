import io
import pyarrow.dataset as ds
import gcsfs

# Initialize GCS file system
fs = gcsfs.GCSFileSystem()

# Open the file from GCS using gcsfs and read with Scanpy
for i in range(1,15):
	if i != 7:
		plate_id = f"plate{i}"
		file_path = f"{plate_id}_annData.h5ad"
		test_plate = f"arc-ctc-tahoe100/2025-02-25/h5ad/{plate_id}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
		try:
			print("Downloading from link:", test_plate)
			fs.get(rpath=test_plate, lpath=file_path)
			print(f"Plate {i} data downloaded..")
		except Exception as e:
			print(f"? Failed: {e}")


