"""Fast, memory-conscious RDS data.frame reader for Python."""

from ._core import (
    InvalidRDS,
    RDSCatalogError,
    RDSError,
    RDSLimitError,
    ReaderLimits,
    UnsupportedRDS,
)
from .api import (
    ParquetTable,
    RDSCatalog,
    RFileInfo,
    RTableInfo,
    convert_rds,
    extract_rds_tables,
    inspect_r_file,
    list_rds_tables,
    materialize_uncompressed,
    read_r_object,
    read_rds,
    read_rds_dataframe,
    to_parquet,
)

__all__ = [
    "InvalidRDS",
    "ParquetTable",
    "RDSCatalog",
    "RDSCatalogError",
    "RDSError",
    "RDSLimitError",
    "RFileInfo",
    "RTableInfo",
    "ReaderLimits",
    "UnsupportedRDS",
    "convert_rds",
    "extract_rds_tables",
    "inspect_r_file",
    "list_rds_tables",
    "materialize_uncompressed",
    "read_r_object",
    "read_rds",
    "read_rds_dataframe",
    "to_parquet",
]

__version__ = "0.4.0a7"
