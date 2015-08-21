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

## How to use

```python
import text_attribution

a = text_attribution.TextAttribution.new_text_attribution_processor(N=4)
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
import attribute_revision
import os

def attribute_revision_for_page(page_id, revision_id, new_revision, new_revision_time):
    filename = '/path/to/page_%d' % page_id
    if os.path.isfile(filename):
        a = text_attribution.TextAttribution.from_json(open(filename).read())
    else:
        a = text_attribution.new_text_attribution_processor(N=4)
    a.add_revision(new_revision.split(), revision_info=str(revision_id),
                   revision_time=new_revision_time)
    attribution = a.get_attribution()
    open(filename, 'w').write(a.to_json()).close()
    return attribution
```

Please see the code for the docstrings of the various methods for detailed documentation.

## Other reading

See also https://etherpad.wikimedia.org/p/mwpersistence for other algorithms.

