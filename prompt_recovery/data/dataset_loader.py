"""
DatasetLoad

Supports:
- Skytrax airline reviewsDataset
- CMS HealthcareDataset (MedAlpaca medical_meadow)
- ECHR European Court of Human Rights casesDataset
- WikiText Dataset
-  CSV/JSON/JSONL Dataset
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Sample"""
    idx: int
    text: str
    metadata: Dict


class DatasetLoader:
    """DatasetLoad"""
    
    def __init__(
        self,
        dataset_path: str,
        text_column: str = "content",
        max_samples: Optional[int] = None,
    ):
        """
        DatasetLoad
        
        Args:
            dataset_path: Dataset
            text_column: text column name
            max_samples: Sample(None = all)
        """
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.max_samples = max_samples
        
        self.data = None
        self.samples = []
    
    def load(self) -> "DatasetLoader":
        """Load dataset"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset: {self.dataset_path}")
        
        logger.info(f"Load dataset: {self.dataset_path}")
        
        # Select the loader by file extension
        ext = os.path.splitext(self.dataset_path)[1].lower()
        
        if ext == '.csv':
            self.data = pd.read_csv(
                self.dataset_path,
                engine='python',
                on_bad_lines='skip'
            )
        elif ext == '.json':
            self.data = pd.read_json(self.dataset_path)
        elif ext == '.jsonl':
            self.data = pd.read_json(self.dataset_path, lines=True)
        else:
            raise ValueError(f"Supports: {ext}")
        
        # Validate that the text column exists
        if self.text_column not in self.data.columns:
            available = list(self.data.columns)
            raise ValueError(f" '{self.text_column}' does not exist; available columns: {available}")
        
        # Sample
        if self.max_samples is not None:
            self.data = self.data.head(self.max_samples)
        
        # Sample
        self.samples = []
        for idx, row in self.data.iterrows():
            text = str(row[self.text_column])
            if text and text != 'nan':
                metadata = {k: v for k, v in row.items() if k != self.text_column}
                self.samples.append(Sample(idx=idx, text=text, metadata=metadata))
        
        logger.info(f"Loadcompleted: {len(self.samples)} Sample")
        return self
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]
    
    def get_texts(self) -> List[str]:
        """Get all texts"""
        return [s.text for s in self.samples]


class SkytraxLoader(DatasetLoader):
    """Skytrax airline reviewsDatasetLoad"""
    
    def __init__(
        self,
        data_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        dataset_path = os.path.join(data_dir, "skytrax_airline_63.csv")
        super().__init__(
            dataset_path=dataset_path,
            text_column="content",
            max_samples=max_samples,
        )


class CMSLoader(DatasetLoader):
    """CMS HealthcareDatasetLoad (MedAlpaca medical_meadow)

    Run prepare_datasets.py first to generate cms_medical.csv.
    """

    def __init__(
        self,
        data_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        dataset_path = os.path.join(data_dir, "cms_medical.csv")
        super().__init__(
            dataset_path=dataset_path,
            text_column="content",
            max_samples=max_samples,
        )


class ECHRLoader(DatasetLoader):
    """ECHR European Court of Human Rights casesDatasetLoad

    Run prepare_datasets.py first to generate echr_cases.csv.
    """

    def __init__(
        self,
        data_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        dataset_path = os.path.join(data_dir, "echr_cases.csv")
        super().__init__(
            dataset_path=dataset_path,
            text_column="content",
            max_samples=max_samples,
        )


class PrivatePIILoader(DatasetLoader):
    """Private-PII DatasetLoad

    Run prepare_datasets.py first to generate private_pii.csv.
    """

    def __init__(
        self,
        data_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        dataset_path = os.path.join(data_dir, "private_pii.csv")
        super().__init__(
            dataset_path=dataset_path,
            text_column="content",
            max_samples=max_samples,
        )


class WikiTextLoader(DatasetLoader):
    """WikiText DatasetLoad"""
    
    def __init__(
        self,
        data_dir: str = "data_cache",
        max_samples: Optional[int] = None,
    ):
        dataset_path = os.path.join(data_dir, "wikitext.csv")
        super().__init__(
            dataset_path=dataset_path,
            text_column="text",
            max_samples=max_samples,
        )


def get_dataset_loader(
    dataset_name: str,
    data_path: Optional[str] = None,
    text_column: str = "content",
    max_samples: Optional[int] = None,
    data_dir: str = "data_cache",
) -> DatasetLoader:
    """
    DatasetLoad
    
    Args:
        dataset_name: Dataset ('skytrax', 'cms', 'echr', 'wikitext', 'custom')
        data_path: custom data path( custom )
        text_column: text column name
        max_samples: Sample
        data_dir: 
        
    Returns:
        DatasetLoad
    """
    if dataset_name == "skytrax":
        return SkytraxLoader(data_dir=data_dir, max_samples=max_samples)
    elif dataset_name == "cms":
        return CMSLoader(data_dir=data_dir, max_samples=max_samples)
    elif dataset_name == "echr":
        return ECHRLoader(data_dir=data_dir, max_samples=max_samples)
    elif dataset_name == "private_pii":
        return PrivatePIILoader(data_dir=data_dir, max_samples=max_samples)
    elif dataset_name == "wikitext":
        return WikiTextLoader(data_dir=data_dir, max_samples=max_samples)
    elif dataset_name == "custom":
        if data_path is None:
            raise ValueError("Dataset data_path")
        return DatasetLoader(
            dataset_path=data_path,
            text_column=text_column,
            max_samples=max_samples,
        )
    else:
        # Try loading as a file path
        return DatasetLoader(
            dataset_path=dataset_name,
            text_column=text_column,
            max_samples=max_samples,
        )
