# authorship-tracking

This code implements the algorithms for tracking the authorship of text in revisioned content
that have been published in WWW 2013: http://www2013.wwwconference.org/proceedings/p343.pdf

The idea consists in attributing each portion of text to the earliest revision where it appeared.
For instance, if a revision contains the sentence "the cat ate the mouse", and the sentence is
deleted, and reintroduced in a later revision (not necessarily as part of a revert), once
re-introduced it is still attributed to its earliest author.

Precisely, the algorithm takes a parameter N.
If a sequence of tokens of length equal or greater than N has appeared before, it
is attributed to its earliest occurrence.  See the paper for details.

The code works by building a trie-based representation of the whole history of the
revisions, in an object of the class AuthorshipAttribution.
Each time a new revision is passed to the object, the object updates its internal state
and it computes the earliest attribution of the new revision, which can be then easily obtained.
The object itself can be serialized (and de-serialized) using json-based methods.

To avoid the representation of the whole past history from growing too much, we remove
from the object the information about content that has been absent from revisions
(a) for at least 90 days, and (b) for at least 100 revisions.  These are
configurable parameters.  With these choices, for the Wikipedia,
the serialization of the object has size typically between 10 and 20 times the size of a
typical revision, even for pages with very long revision lists.  See paper for detailed
experimental results.

## How to use

```python
import authorship_attribution

a = authorship_attribution.AuthorshipAttribution.new_attribution_processor(N=4)
a.add_revision("I like to eat pasta".split(), revision_info="rev0")
a.add_revision("I like to eat pasta with tomato sauce".split(), revision_info="rev1")
a.add_revision("I like to eat rice with tomato sauce".split(), revision_info="rev3")
print a.get_attribution()
```
produces:
```
['rev0', 'rev0', 'rev0', 'rev0', 'rev3', 'rev1', 'rev1', 'rev1']
```

It is possible to serialize and de-serialize the text attribution object using json.
In this way, the text attribution object can be stored (for instance, together with the
wiki page it is intended to analyze).  Each time a new revision needs to be
attributed, one can read the serialized text attribution object,
deserialize it, pass it the new revision, re-serialize it and finally write it back:

```python
import authorship_attribution
import os

def attribute_revision_for_page(page_id, revision_id, new_revision, new_revision_time):
    filename = '/path/to/page_%d' % page_id
    if os.path.isfile(filename):
        a = authorship_attribution.AuthorshipAttribution.from_json(open(filename).read())
    else:
        a = authorship_attribution.new_attribution_processor(N=4)
    a.add_revision(new_revision.split(), revision_info=str(revision_id),
                   revision_time=new_revision_time)
    attribution = a.get_attribution()
    open(filename, 'w').write(a.to_json()).close()
    return attribution
```

Please see the code for the docstrings of the various methods for detailed documentation.

## Other reading

See also https://etherpad.wikimedia.org/p/mwpersistence for other algorithms.

