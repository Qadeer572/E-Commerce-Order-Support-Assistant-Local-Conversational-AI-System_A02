import pytest
from backend.conversation_manager.manager import source_footer
from backend.retrieval.types import Hit


def test_only_included_sources_are_deduplicated():
    hits=[Hit(str(i),'one.md','One','facts',.9,'product') for i in range(2)]
    hits.append(Hit('other','other.md','Other','discarded',.9,'product'))
    footer=source_footer(hits,'Retrieved evidence [one.md] One:\nfacts\nEnd evidence.')
    assert footer.count('one.md')==1
    assert 'other.md' not in footer


def test_no_evidence_has_no_footer():
    assert source_footer([], 'greeting')==''
