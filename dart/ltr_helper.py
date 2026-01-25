from joblib import Parallel, delayed
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dart.common import *

def qid_key(ua: UserActivity):
    return (ua.query, ua.issuer, ua.timestamp)

def write_records(export_path: str, roles: dict[str, list[ClickThroughRecord]]):
    os.makedirs(export_path, exist_ok=True)
    for role, records in roles.items():
        with open(f"{export_path}/{role}.txt", "w") as f:
            f.writelines(str(record) + "\n" for record in records)

class LTRDatasetMaker:
    def __init__(self, activities: list[UserActivity]):
        self.activities = activities
        self.qid_mappings = {qid_key(ua) for ua in self.activities}
    
    @property
    def qid_mappings(self):
        return self._qid_mappings

    @qid_mappings.setter
    def qid_mappings(self, value):
        """
        Sorted for consistent qid assignments in parallel processing.
        """
        self._qid_mappings = tuple(sorted(value))

    def generate(self, export_path: str, normalize: bool = True):
        records = self.compile_records()
        train_records, vali_records, test_records = split_dataset_by_qids(records)

        write_records(export_path, {
            "train": train_records, 
            "vali": vali_records, 
            "test": test_records
        })

        if normalize:
            normalize_features(export_path)

    def write_queries(self, export_path: str):
        with open(f"{export_path}/queries.tsv", "w") as f:
            for qid, (query, user, timestamp) in enumerate(self.qid_mappings):
                f.write(f"qid:{qid}\t{query}\t{user}\t{timestamp}\n")

    def compile_records(self) -> list[ClickThroughRecord]:

        def process_row(ua: UserActivity):
            qid = self.qid_mappings.index((ua.query, ua.issuer, ua.timestamp))
            query_terms = tokenize(ua.query)

            # Build corpus from only the documents in this query's result list
            local_corpus = {
                doc.infohash: doc.torrent_info.title.lower()
                for doc in ua.results
            }
            corpus = Corpus(local_corpus)

            row_records = []
            for result in ua.results:
                record = ClickThroughRecord()
                record.qid = qid
                record.rel = int(result.infohash == ua.chosen_result.infohash)

                v = QueryDocumentRelationVector()
                v.title = corpus.compute_features(result.infohash, query_terms)
                v.seeders = min(result.seeders, 10000)
                v.leechers = min(result.leechers, 10000)
                v.age = ua.timestamp - result.torrent_info.timestamp
                v.pos = result.pos
                v.size = result.torrent_info.size
                v.entropy = calculate_shannon_entropy(result.torrent_info.title)
                v.diversity = calculate_trigram_diversity(result.torrent_info.title)

                match = re.search(r'\b(19|20)\d{2}\b', result.torrent_info.title)
                if match:
                    title_year = int(match.group(0))
                    upload_year = datetime.fromtimestamp(result.torrent_info.timestamp).year
                    current_year = datetime.fromtimestamp(ua.timestamp).year
                    content_age = current_year - title_year
                    if content_age > 0:
                        v.content_age = np.log1p(content_age)
                    temporal_gap = title_year - upload_year
                    if temporal_gap > 0:
                        v.temporal_gap = np.log1p(temporal_gap)

                
                record.qdr = v
                row_records.append(record)

            return row_records

        parallel_records = Parallel(n_jobs=1, batch_size=8)(
            delayed(process_row)(ua)
            for ua in tqdm(self.activities, desc="Compiling records")
        )

        records = [record for batch in parallel_records for record in batch]
        
        return records
    