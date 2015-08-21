#!/usr/bin/python
# Michael Shavlovsky, 2012
# Luca de Alfaro, 2015
# BSD License.

import datetime
import json_plus
import json
import os
import numpy as np
import unittest


class AuthorshipAttribution(json_plus.Serializable):
    """Usage (see also unit tests at the end):
    a = TextAttribution.new_text_attribution_processor(N=4)
    a.add_revision(list_of_tokens1, revision_info="rev0")
    a.add_revision(list_of_tokens1, revision_info="rev0")
    a.add_revision(list_of_tokens1, revision_info="rev0")
    print a.get_attribution()
    s = a.to_json()
    b = TextAttribution.from_json(s)
    print b.get_attribution()
    """

    def __init__(self):
        """The initializer is empty, as required by json_plus."""
        pass

    def initialize(self, N=4, K_revisions=100, K_time=None, will_truncate=True,
                   dummy_token=u'banana'):
        """Initializes the text attribution processing.  Call ONCE after
        instantiating a new object, or just use the create_text_attribution_processor
        method.
        * The parameter N is such that every repetition of at least N input tokens is rare,
          and is likely to denote copying rather than original content.  That is, if a
          subsequent author adds some content that contains N or more consecutive tokens from a
          previously existing revision, the tokens are attributed to the previously
          existing revision, not to the author.
        * The trie guarantees tracking of at least K_revisions for K_time (default: 90 days).
        * If will_truncate is True, the trie will truncate content that is dead (not in the
          latest revision), and both older than K_revisions and K_time.
        * dummy_token is a token that is guaranteed not to occur in any actual revision.
          It must be json-serializable back and forth onto itself.
        """
        self.N = N
        self.K_revisions = K_revisions
        self.K_time = K_time or datetime.timedelta(days=90)
        self.will_truncate = will_truncate
        self.dummy_token = dummy_token

        # initialization class constants

        # index of last revision when a node was visited
        # by the algorithm
        self.NODE_LAST_VISIT = 0;
        # index of the covering label in a leaf
        # the covering label is either an integer or a list
        # it is integer in case all tokens have
        # the same label and a list otherwise
        self.NODE_COVERING_LABEL = 1
        # source node id of an edge
        self.EDGE_SRC_NODE_ID = 0
        # destination node id of an edge
        self.EDGE_DEST_NODE_ID = 1
        # tokens which are on an edge
        self.EDGE_TOKENS = 2
        # Minimum number of days before things are put
        # into the list subject to time pruning, for efficiency.
        self.MINDAYS = datetime.timedelta(days=5)

        # initializing class data members
        self.nodes = {}
        self.edges = {}
        # last inserted node id
        self.lin_id = -1
        # timeTable is a mapping revisions <=> timestamps
        self.timetable = []
        # timestamp of current revision.
        self.cr_timestamp = None
        # cr_number is current revision number
        self.cr_number = -1
        # cr_covering is covering for current revision, i.e.
        # self.cr_covering[i] is an author of token i in
        # current revision (self.crevision)
        self.cr_covering = []
        # crevision is current revision - list of tokens
        self.crevision = []
        # revision_info is information on each revision (e.g., revision id, or author, ...)
        self.revision_info = []
        # inserting root node.
        self._add_node()
        # revision_id_table is a mapping wiki revision id <=> index number
        # the table have similar role as timetable
        self.revision_id_table = []

    @staticmethod
    def new_attribution_processor(N=4, K_revisions=100, K_time=None,
                                       will_truncate=True, dummy_token=u'\x00'):
        """Convenience function for creating a text attribution object.
        Just do:
        attributor = TextAttribution.create_text_attribution_processor(N=4)
        (for example).
        """
        a = AuthorshipAttribution()
        a.initialize(N=N, K_revisions=K_revisions, K_time=K_time,
                     will_truncate=will_truncate, dummy_token=dummy_token)
        return a

    def add_revision(self, revision, timestamp=None, revision_info=None):
        """This method adds a new revision to the analysis.
        * The revision is an arbitrary list of tokens, where the only requirement
          on a token is that it be json_serializable.
        * The timestamp is used to decide when to prune information for
          text that has long been dead, so use it (see parameters
          K_time and will_truncate).  Defaults to current time.
        * revision_info is an arbitrary json-serializable piece of information about
          the revision.  It can consist of the revision id, or in the json dump of a
          dictionary containing e.g. revision id, revision author, revision timestamp,
          and more.  This information will be returned as attribution for the text.
          It defaults to the ordinal number of the revision.
        """
        self.crevision = [self.dummy_token] * self.N + revision[:] + [self.dummy_token] * self.N
        self.cr_number += 1
        if revision_info is None:
            revision_info = str(self.cr_number)
        self.revision_info.append(revision_info)
        # last inserted revision id
        # todo(michael): next line changed in a hurry
        #self.last_inserted_revision_id = wiki_revision_id
        self.last_inserted_revision_id = self.cr_number
        # saving previous timestamp
        self.cr_timestamp = timestamp or datetime.datetime.utcnow()
        self._update_timetable(timestamp)
        self._update_revision_id_table()
        # calculating covering and adding nodes to the tree if needed
        # list_of_leaves_id is a list such that i-th element of it
        # is a node id of a leaf with string label self.crevision[i : i + N]
        list_of_leaves_id = self._calculate_covering_and_add_branches()
        # update covering labels
        self._update_covering_labels(list_of_leaves_id)
        # truncating the tree
        if self.will_truncate:
            th = self._get_threshold()
            self._truncate(th)
        # To save space in serialization.
        self.crevision = None


    def get_attribution(self):
        """This function can be called after add_revision, and it is used to
        read the origin of every token in the last revision.
        It returns a list of information, one for each token of the last revision.
        Each piece of information corresponds to the revision_info of the revision
        where the token was inserted.
        """
        idxs = self.cr_covering[self.N : -self.N]
        return [self.revision_info[x] for x in idxs]


    def get_attribution_abbrev(self):
        """Similar to the above function, but returns:
        - a list of ids, one per token.
        - a mapping from ids to revision_info.
        The advantage of this function is that, since many tokens commonly
        originate from the same revisions, if revision_info is large, it
        produces much smaller output."""
        idxs = self.cr_covering[self.N : -self.N]
        infos = {i: self.revision_info[i] for i in idxs}
        return idxs, infos

    @staticmethod
    def from_json(s):
        obj = json_plus.Serializable.from_json(s)
        # Unfortunately, keys in json become strings.  So we have to fix some dictionaries.
        obj.nodes = {int(i): x for i, x in obj.nodes.items()}
        obj.edges = {int(i): x for i, x in obj.edges.items()}
        return obj

    ### Internal methods below this point.

    def _add_node(self):
        """ method adds node to array self.nodes
        returns an id of a new node
        """
        new_node_id = self._get_new_node_id()
        new_node = []
        # setting node last visit to current revision
        new_node.append(self.cr_number)
        self.nodes[new_node_id] = new_node
        return new_node_id

    def _get_node(self, node_id):
        self.nodes[node_id][self.NODE_LAST_VISIT] = self.cr_number
        return self.nodes[node_id]

    def _get_leaf_covering_label(self, leaf_id):
        """ method returns covering label of
        a node with id leaf_id
        if the label is an integer then we return list of length N
        filled with the integer
        """
        labels = self.nodes[leaf_id][self.NODE_COVERING_LABEL]
        if not isinstance(labels, list):
            labels = [labels] * self.N
        return labels

    def _set_covering_label(self, leaf_id, new_label):
        """ method sets covering label for a leaf with id leaf_id,
        new_label can be an integer of a list
        """
        node = []
        # adding last visit
        node.append(self.cr_number)
        # adding covering label
        node.append(new_label)
        self.nodes[leaf_id] = node

    def _add_edge(self, src_node_id, dest_node_id, tokens):
        if not self.edges.has_key(src_node_id):
            self.edges[src_node_id] = {}
        new_edge = []
        new_edge.append(src_node_id)
        new_edge.append(dest_node_id)
        new_edge.append(tokens[:])
        self.edges[src_node_id][tokens[0]] = new_edge

    def _get_edge(self, src_node_id, first_token):
        e = self.edges[src_node_id][first_token]
        # remember last visit
        self._get_node(src_node_id)
        self._get_node(e[self.EDGE_DEST_NODE_ID])
        return e

    def _edge_exist(self, src_node_id, first_token):
        """ method returns true if the edge exists in the tree
        """
        return (self.edges.has_key(src_node_id) and
                    self.edges[src_node_id].has_key(first_token))

    def _split_edge(self, src_node_id, first_token, split_point):
        """ src_node_id and first_token determines the edge to split
        split_point is lenth of tokens in firs part of the edge
        method returns id of a new node that splits old edge
        """
        e = self._get_edge(src_node_id, first_token)
        # shrinking the edge
        e_dest_node_id_old = e[self.EDGE_DEST_NODE_ID]
        middle_node_id = self._add_node()
        first_part_of_e = e[self.EDGE_TOKENS][:split_point]
        second_part_of_e = e[self.EDGE_TOKENS][split_point:]
        self._add_edge(src_node_id, middle_node_id, first_part_of_e)
        # creating 2nd part of the edge
        self._add_edge(middle_node_id, e_dest_node_id_old, second_part_of_e)
        return middle_node_id

    def _calculate_covering_and_add_branches(self):
        """ method calculates covering for current
        revision (sets property self.cr_covering)
        and adds new branches to the tree
        (through the method get_leaf_nodeid_or_add_it)
        method returns list_of_leaves_id which is a list such
        that i-th element of it is a node id of a leaf with
        string label self.crevision[i : i + N]
        """
        # to calculate covering for each token in current revision
        # we build matrix matrix_labels such that
        # for i-th token coresponds matrix_lables[i] wich is an array of labels
        # for substring self.crevision[i],...,self.crevision[i + N - 1]
        # last N-1 indices are filled with self.cr_number
        matrix_labels = np.zeros( (len(self.crevision), self.N) )
        matrix_labels = matrix_labels + self.cr_number
        # list_of_leaves_id[idx] is a leaf id with string label
        # self.crevision[idx:idx + N]
        list_of_leaves_id = []
        # case when revision is too short
        if len(self.crevision) < self.N :
            self.cr_covering = [self.cr_number] * len(self.crevision)
            return list_of_leaves_id
        # filling matrix_labels
        for i in xrange(len(self.crevision) - self.N + 1):
            s = self.crevision[i: i + self.N]
            leaf_id = self._get_leaf_nodeid_or_add_it(s)
            list_of_leaves_id.append(leaf_id)
            matrix_labels[i] = self._get_leaf_covering_label(leaf_id)
        # calculating covering
        covering = []
        # take care of first N-1 tokens
        for i in xrange(self.N - 1):
            val = matrix_labels[i][0]
            for k in xrange(i, -1, -1):
                val = min(val, matrix_labels[k][i - k])
            covering.append(int(val))
        # rest of tokens
        for i in xrange(self.N-1, len(self.crevision), 1):
            val = matrix_labels[i][0]
            for k in xrange(self.N):
                val = min(val, matrix_labels[i-k][k])
            covering.append(int(val))
        self.cr_covering = covering
        return list_of_leaves_id

    def _get_leaf_nodeid_or_add_it(self, s):
        """ given string s (list of tokens) of length N
        the method returns leaf id that string path
        the root -> the leaf equals s
        If there is no such leaf, then
        method add it to the tree and sets
        covering label of the leaf as current revision number
        """
        # we need to "match" substring s along the tree
        # trying to make step down the tree
        # from node with current_node_id
        # set it to the root in the beginning
        # if we cannot make step then add
        # new edge so the tree contains string s
        current_node_id = 0
        # we are trying match substring s[token_idx] ... s[end]
        # in the beginning token_idx = 0
        token_idx = 0;
        while True:
            rev_tokens = s[token_idx:]
            # if the edge along which we need
            # to traverse down the tree
            # doesn not exist then we create
            # new edge and return new leaf id
            if not self._edge_exist(current_node_id, rev_tokens[0]):
                new_node_id = self._add_node()
                self._add_edge(current_node_id, new_node_id, rev_tokens)
                # adding covering label to the new created leaf
                self._set_covering_label(new_node_id, self.cr_number)
                return new_node_id
            # we can traverse down
            e = self._get_edge(current_node_id, rev_tokens[0])
            e_tokens = e[self.EDGE_TOKENS]
            # now compare tokens from the edge (e_tokens)
            # and rev_tokens (the rest of string that we need to add
            # to the tree)
            prefix_len = self._length_common_prefix(e_tokens, rev_tokens)

            # if $revTokens fits on the edge e, then
            # nothing to do, just return
            if prefix_len == len(rev_tokens):
                return e[self.EDGE_DEST_NODE_ID]

            # if edge e fits on the rev_tokens
            # then continue traversing
            if prefix_len == len(e_tokens):
                token_idx += len(e_tokens)
                current_node_id = e[self.EDGE_DEST_NODE_ID]
                continue

            # now we need create new edge because
            # prefix_len is less than lenght of the edge
            middle_node_id = self._split_edge(current_node_id,
                                             rev_tokens[0],
                                             prefix_len)
            # creating new edge with rev_tokens
            new_node_id = self._add_node()
            self._add_edge(middle_node_id,
                          new_node_id,
                          rev_tokens[prefix_len:])
            # adding covering label to the new created leaf
            self._set_covering_label(new_node_id, self.cr_number)
            return new_node_id

    def _update_covering_labels(self, list_of_leaves_id):
        """  method updates all covering labels in the tree
        given covering for current revision (property self.cr_covering)
        and list_of_leaves_id (resul of calculate_covering_and_add_branches)
        """
        for i in xrange(len(self.crevision) - self.N + 1):
            leaf_id = list_of_leaves_id[i]
            labels = self.cr_covering[i : i + self.N]
            # if values in the list labels are coinside then we
            # store only the integer
            if min(labels) == max(labels):
                labels = labels[0]
            # setting covering label
            self._set_covering_label(leaf_id, labels)

    def _update_revision_id_table(self):
        """ revision id timetable is needed for
        truncating nodes of the tree.
        timetable is a list of tuples
        (revision index number, self.cr_number)
        add new elemnts to the end
        [oldest ----- most recent]
        """
        # adding rule: if distance between
        # current revsion and previous inserted revision
        # is more or equal than minrev revisions then add new tuple
        # to the list
        # todo(Michael): make minrev as an argument or constant
        minrev = 5
        if len(self.revision_id_table) == 0:
            tuple2add = (self.cr_number, self.cr_number)
            self.revision_id_table.append(tuple2add)
            return
        revdist = self.cr_number - self.revision_id_table[-1][0]
        if revdist >= minrev:
            tuple2add = (self.cr_number, self.cr_number)
            self.revision_id_table.append(tuple2add)
        # deleting rule: delete tuple from the table
        # if distance between current revision and tuple is more than
        # max(self.K_rev, number_of_revisions_in_timetable) + minrev revisions
        if len(self.timetable) == 0:
            th = 0
        else:
            inc_number_oldest = self.timetable[0][0]
            th = self.cr_number - inc_number_oldest
        th = max(self.K_revisions, th) + minrev
        while True:
            revdist = self.cr_number - self.revision_id_table[0][0]
            if revdist > th:
                del self.revision_id_table[0]
            else:
                break

    def _update_timetable(self, timestamp=None):
        """ time table is needed for
        truncating nodes of the tree.
        timetable is a list of tuples
        (revision index number, timestamp)
        add new elements to the end
        [oldest ----- most recent]
        """
        self.cr_timestamp = timestamp
        if timestamp == None:
            return
        # adding rule: if day distance between
        # current revision and previous inserted revision
        # is more or equal than MINDAYS then add new tuple
        # to the time list
        if len(self.timetable) == 0:
            tuple2add = (self.cr_number, self.cr_timestamp)
            self.timetable.append(tuple2add)
            return
        if self.cr_timestamp - self.timetable[-1][1] > self.MINDAYS:
            tuple2add = (self.cr_number, self.cr_timestamp)
            self.timetable.append(tuple2add)
        # deleting rule: delete tuple from the timetable
        # if distance between current revision and tuple is
        # more thatn self.K_time+mindays days
        while True:
            if self.cr_timestamp - self.timetable[0][1] > self.K_time:
                del self.timetable[0]
            else:
                break

    def _get_threshold(self):
        """ method calculates threshold for truncating the tree.

        note that revision_id_table stores maximum between K_rev revisions and
        number of revisions stored in the timetable, so threshold is the oldest
        revision id stored in revision_id_table
        """
        return self.revision_id_table[0][1]

    def _truncate(self, threshold):
        """ the function delete nodes and edges.
        all nodes have revision id when they were visited last time.
        nodes and edges which were visited last time
        before threshold will be deleted.
        """
        # todo(michael): rewrite function so it is more readable.
        # sorted list of pairs - id and how many
        # revisions ago it was visited
        # sorted by second parameter - revisions between last visit
        threshold = self.cr_number - threshold
        #print 'real threshold is ', threshold
        l = [(x, self.cr_number - y[self.NODE_LAST_VISIT]) for x, y
                              in sorted(self.nodes.iteritems(),
                                        key = lambda (x, y): self.cr_number
                                            - y[self.NODE_LAST_VISIT])]
        lv = np.array([y for x,y in l])
        #print 'lv', lv
        temp = 2 * threshold
        #idx = lv <= threshold
        idx = lv < threshold
        #print 'idx', idx
        lv[idx] = temp
        #print 'lv', lv
        idx_min = lv.argmin()
        if lv[idx_min] == temp:
            return
        id_to_del = [x for x,y in l[idx_min:]]
        #print 'id to del ',id_to_del
        # deleting
        for x in id_to_del:
            if self.edges.has_key(x):
                del self.edges[x]
        # deleting edges when rare node is
        # destination node
        edges_id_to_del = []
        for x in self.edges:
            for y in self.edges[x]:
                e = self.edges[x][y]
                if e[self.EDGE_DEST_NODE_ID] in id_to_del:
                    edges_id_to_del.append((x,y))
        for x,y in edges_id_to_del:
            del self.edges[x][y]
        for x in id_to_del:
            del self.nodes[x]

    def _get_new_node_id(self):
        self.lin_id += 1
        return self.lin_id

    def _length_common_prefix(self, str1, str2):
        """ returns length of maximal common
        prefix between arguments
        """
        res = 0
        for i in xrange(min(len(str1), len(str2))):
            if str1[i] != str2[i]:
                return res
            res += 1
        return res

    def is_leaf(self, node_id):
        return node_id not in self.edges


def plot_attribution_tree(nodes, edges, revisions, file_name):
    """
    This is just plotting code, and it is not necessary for the
    functioning of the authorship tracking.
    file_name is a name of a file to save image
    nonleaf nodes have text: node_id(node last visit)
    leaf nodes have text: [covering label](node last visit)
    """
    NODE_LAST_VISIT = 0;
    NODE_COVERING_LABEL = 1;
    EDGE_SRC_NODE_ID = 0;
    EDGE_DEST_NODE_ID = 1;
    EDGE_TOKENS = 2;
    dot_file_name = "plot_trie.dot"
    colors = ['black', 'blue', 'red',
              'forestgreen', 'darkorange',
              'chartreuse', 'deeppink1',
              'firebrick4', 'darkviolet',
              'brown', 'chocolate4']
    # creating nodes names
    node_names = {}
    for x in nodes.keys():
        n = x
        if x < 0:
            n = 10000 + x
        s = "node_%s"%(n)
        node_names[x] = s
    # creating edges
    edge_lines = []
    for x in edges:
        e = edges[x]
        for y in e:
            s = ("%s -> %s [label=\"%s\"];\n"%
                (node_names[e[y][EDGE_SRC_NODE_ID]],
                 node_names[e[y][EDGE_DEST_NODE_ID]],
                 ' '.join(e[y][EDGE_TOKENS])))
            edge_lines.append(s)
    # creating nodes
    node_lines = []
    for x in nodes:
        n = nodes[x]
        label = "%s"%(x)
        if len(nodes[x]) == 2:
            cov_label = nodes[x][NODE_COVERING_LABEL]
            if isinstance(cov_label, list):
                cov_label = ' '.join([str(z) for z in cov_label])
            label = '[%s]' % cov_label
            label = '%s(%s)' % (label, nodes[x][NODE_LAST_VISIT])
        else:
            label = '%s(%s)' % (label, nodes[x][NODE_LAST_VISIT])
        s = "%s [label=\"%s\"];\n" % (node_names[x], label)
        node_lines.append(s)
    # revision lines
    revision_lines = []
    for i in xrange(len(revisions)):
        s = ("revision_%s [shape=plaintext, label=\"%s:%s\"];\n" %
             (i, i, ' '.join(revisions[i])))
        revision_lines.append(s)
    # writing sript
    prog = "digraph G {\n"
    for x in revision_lines:
        prog = prog + x
    for x in node_lines:
        prog = prog + x
    for x in edge_lines:
        y  = x.encode('utf-8')
        prog = '%s%s'%(prog, y)
    prog = prog + "}\n"
    dot_file = 'plot_tree.dot'
    f = open(dot_file,'w')
    f.write(prog)
    f.close()
    temp = 'dot -Tpdf -Gcharset=latin1 %s -o %s'%(dot_file, file_name)
    print temp
    os.system(temp)

# Internal information (do not read if you just want to use this):
#
# Class TriesA5 stores all data and methods
# necessary for constructing a tree for A5 algorithm
# Each leaf represents some string s obtained
# by traversing the tree from root to the leaf.
# In each leaf we store covering lables of
# tokens in the string s on the path the root -> the_leaf.
#
# information about nodes is stored in a dictionary self.nodes
# self.nodes[nodeId] returns an list, say, the_node which
# stores infromation about a node, indices below
# which starts from NODE_ are indices in list the_node
#
# information about edges is stored in a dictionary self.edges
# key is souce node id and value is a dictionary, say, edge_group
# key of edge_group is first token and value is a list which contains
# source node id, destination node id, and tokens.
# So an edge is retreived by its souce node id and first token:
# e = self.edges[source_node_id][first_token]


class TestTextAttribution(unittest.TestCase):

    def test_one_revision(self):
        trie = AuthorshipAttribution()
        trie.initialize(N=4)
        trie.add_revision('I like cats and dogs'.split(), revision_info=0)
        # print "One:", trie.get_attribution()
        self.assertEqual(trie.get_attribution(), [0, 0, 0, 0, 0])

    def test_two_revisions(self):
        trie = AuthorshipAttribution()
        trie.initialize(N=4)
        trie.add_revision('I like cats and dogs'.split(), revision_info=0)
        trie.add_revision('I like cats and dogs but even more birds'.split(), revision_info=1)
        # print "Two", trie.get_attribution()
        # self.assertEqual(trie.cr_covering, [0, 0, 0, 0, 0])

    def test_three_revisions(self):
        trie = AuthorshipAttribution.new_attribution_processor(N=4)
        trie.add_revision('I like cats and dogs'.split(), revision_info='luca')
        trie.add_revision('I like cats and dogs but even more birds'.split(), revision_info='matt')
        trie.add_revision('I like cats and elephants but even more birds'.split(), revision_info='george')
        # print "Three:", trie.get_attribution()
        self.assertEqual(trie.get_attribution(), ['luca', 'luca', 'luca', 'luca',
                                                'george', 'matt', 'matt', 'matt', 'matt'])
        # print trie.get_attribution_abbrev()
        idxs, info = trie.get_attribution_abbrev()
        self.assertEqual(trie.get_attribution(), [info[x] for x in idxs])

    def test_serialization_simple(self):
        a = AuthorshipAttribution.new_attribution_processor(N=4)
        a.add_revision('I like pasta al pomodoro'.split(), revision_info="rev0")
        a.add_revision("I don't like pasta al pomodoro".split(), revision_info="rev1")
        a.add_revision("I like risotto a lot more than pasta al pomodoro".split(), revision_info="rev2")
        # print a.get_attribution()
        s = a.to_json()
        # print s
        b = AuthorshipAttribution.from_json(s)
        # print b.get_attribution()
        self.assertEqual(a.get_attribution(), b.get_attribution())

    def check_equal_edges(self, a, b):
        print "--start check--"
        for k, v in a.edges.items():
            print "  Checkind edges for", k
            print "     a:", a.edges[k]
            print "     b:", b.edges[k]
            for kk, vv in v.items():
                print "         a[%r][%r]" % (k, kk), a.edges[k][kk]
                print "         b[%r][%r]" % (k, kk), b.edges[k][kk]
                self.assertEqual(a.edges[k][kk], b.edges[k][kk])
        print "--end check--"

    def test_serialization_two_steps(self):
        a = AuthorshipAttribution.new_attribution_processor(N=4)
        a.add_revision(u'I like pasta al pomodoro'.split(), revision_info="rev0")
        a.add_revision(u"I don't like pasta al pomodoro".split(), revision_info="rev1")
        a.add_revision(u"I like risotto a lot more than pasta al pomodoro".split(), revision_info="rev2")
        b = AuthorshipAttribution.from_json(a.to_json())
        # print "Nodes a:", a.nodes
        # print "Nodes b:", b.nodes
        # print "Edges a:", a.edges
        # print "Edges b:", b.edges
        self.assertEqual(a.nodes, b.nodes)
        self.assertEqual(a.edges, b.edges)
        # self.check_equal_edges(a, b)
        # self.check_equal_edges(b, a)
        new_rev = 'I like risotto much better than pasta al pomodoro'.split()
        a.add_revision(new_rev, revision_info="rev3")
        b.add_revision(new_rev, revision_info="rev3")
        # print "a attr:", a.get_attribution()
        # print "b attr:", b.get_attribution()
        self.assertEqual(a.get_attribution(), b.get_attribution())

    def make_revision(self, r):
        import random
        tokens = ['a', 'bb', 'ccc']
        def get_tokens(n):
            return [random.choice(tokens) for _ in range(n)]
        k = random.randrange(0, len(r))
        if random.getrandbits(1):
            # insert
            l = random.randrange(2, 8)
            r[k: k + l] = get_tokens(l)
        else:
            # replace
            r[k] = random.choice(tokens)
        return r

    def test_serialization_manytimes(self):
        a = AuthorshipAttribution.new_attribution_processor(N=3)
        r = ['a']
        a.add_revision(r)
        b = AuthorshipAttribution.from_json(a.to_json())
        for _ in range(100):
            r = self.make_revision(r)
            a.add_revision(r)
            b = AuthorshipAttribution.from_json(b.to_json())
            b.add_revision(r)
            self.assertEqual(a.get_attribution(), b.get_attribution())

    def test_doc1(self):
        a = AuthorshipAttribution.new_attribution_processor(N=4)
        a.add_revision("I like to eat pasta".split(), revision_info="rev0")
        a.add_revision("I like to eat pasta with tomato sauce".split(), revision_info="rev1")
        a.add_revision("I like to eat rice with tomato sauce".split(), revision_info="rev3")
        print a.get_attribution()


if __name__ == '__main__':
    unittest.main()
