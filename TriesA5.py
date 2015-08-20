#!/usr/bin/python
# Michael Shavlovsky, 2012

import os
import numpy as np
import json
import cjson
import glob
import wiki2text_v2 as w2t
import re
import time
import gzip


class TriesA5(object):
    """ class TriesA5 stores all data and methods
    necessary for constructing a tree for A5 algorithm
    Each leaf represents some string s obtained
    by traversing the tree from root to the leaf.
    In each leaf we store covering lables of
    tokens in the string s on the path the root -> the_leaf.

    information about nodes is stored in a dictionary self.nodes
    self.nodes[nodeId] returns an list, say, the_node which
    stores infromation about a node, indices below
    which starts from NODE_ are indices in list the_node

    information about edges is stored in a dictionary self.edges
    key is souce node id and value is a dictionary, say, edge_group
    key of edge_group is first token and value is a list which contains
    source node id, destination node id, and tokens.
    So an edge is retreived by its souce node id and first token:
    e = self.edges[source_node_id][first_token]
    """
    def __init__(self, N=4, K_rev=100, K_time=90, will_truncate=True):
        # initialization class constants
        # last revision when a node was visited
        # by the algorithm
        self.NODE_LAST_VISIT = 0;
        # covering label in a leaf
        # covering label is whether integer or a list
        # it is integer in case all tokens have
        # the same label and a list otherwise
        self.NODE_COVERING_LABEL = 1;
        # source node id of an edge
        self.EDGE_SRC_NODE_ID = 0;
        # destination node id of an edge
        self.EDGE_DEST_NODE_ID = 1;
        # tokens which are on an edge
        self.EDGE_TOKENS = 2;

        # initializing class data members
        self.nodes = {}
        self.edges = {}
        # N is a constant with propertie that
        # every string of lenght >= N is rare
        # N determines depth of the tree
        self.N = N
        # last inserted node id
        self.lin_id = -1
        # the tree keeps at least K_revisions revisions
        # and at least K_time days of revisions
        # (revisions that are not older than K_time days)
        self.K_revisions = K_rev
        self.K_time = K_time
        self.will_truncate = will_truncate
        # timeTable is a mapping revisions <=> timestamps
        self.timetable = []
        # timestamp format is 'yyyymmddhhmmss'
        self.cr_timestamp = None
        # previous revision timestamp
        self.pr_timestamp = None
        # cr_number is current revision number
        self.cr_number = -1
        # cr_covering is covering for current revision, i.e.
        # self.cr_covering[i] is an author of token i in
        # current revision (self.crevision)
        self.cr_covering = []
        # crevision is current revision - list of tokens
        self.crevision = []
        # inserting root
        self.add_node()
        # rev_incremental_number is a revision index number
        # e.g. first added revision has rev_incremental_number one,
        #      second added revision has rev_incremental_number two,
        #      third added revision has rev_incremental_number three, etc
        self.rev_incremental_number = 0
        # revision_id_table is a mapping wiki revision id <=> index number
        # the table have similar role as timetable
        self.revision_id_table = []

    @classmethod
    def from_json(cls, json_string):
        """ alternative constructor for creating an object of class TriesA5
        from json string which was obtained by method dump2json
        """
        result = cls()
        result.load_from_json(json_string)
        return result

    def add_revision(self, revision, timestamp=None,
                     user_id=None, user_name=None, wiki_revision_id=None):
        """ methods adds revision r_k (k = 1, 2, 3, ... ) to the tree
        revision is array of tokens
        let's denote S a set of all substrings s from r_k of length N

        If k == 1 then in each leaf we store 1 as covering label.
        If k > 1n then :
             - obtain covering r_k given tree for
               revisions r_1, ..., r_{k-1}
             - add to the tree all subsrings s of r_k of length N
             - update covering labels for all leaves
               with path strings from set S
        """
        # todo(michael): add back dummy tokens
        #revision = self.add_dummy_tokens(revision)

        # user_id, 0 - if the user is anonymous or nonexistent
        # user_name is ip adress if the user is anonymous
        self.user_id = user_id
        self.user_name = user_name
        self.crevision = revision
        if wiki_revision_id != None:
            self.cr_number = wiki_revision_id
        else:
            self.cr_number += 1
        # last inserted revision id
        # todo(michael): next line changed in a hurry
        #self.last_inserted_revision_id = wiki_revision_id
        self.last_inserted_revision_id = self.cr_number
        # revision index number
        self.rev_incremental_number += 1
        # saving previous timestamp
        self.pr_timestamp = self.cr_timestamp
        self.cr_timestamp = timestamp
        # check that new revision is older that prevision one
        # todo(michael): do we need it?
        #if (self.pr_timestamp != None and self.cr_timestamp != None
        #    and (int(self.cr_timestamp) < int(self.pr_timestamp))):
        #    raise Exception('newer revision has already been added')
        self.update_timetable(timestamp)
        self.update_revision_id_table()
        # calculating covering and adding nodes to the tree if needed
        # list_of_leaves_id is a list such that i-th element of it
        # is a node id of a leaf with string label self.crevision[i : i + N]
        list_of_leaves_id = self.calculate_covering_and_add_branches()
        # update covering labels
        self.update_covering_labels(list_of_leaves_id)
        # truncating the tree
        if self.will_truncate:
            th = self.get_threshold()
            self.truncate(th)

    def add_dummy_tokens(self, revision):
        """ method adds N dummy tokens to the beginning and to the end
        of the revision
        """
        dummy = ["dummy"] * self.N
        result = []
        result.extend(dummy)
        result.extend(revision)
        result.extend(dummy)
        return result

    def add_node(self):
        """ method adds node to array self.nodes
        returns an id of a new node
        """
        new_node_id = self.get_new_node_id()
        new_node = []
        # setting node last visit to current revision
        new_node.append(self.cr_number)
        self.nodes[new_node_id] = new_node
        return new_node_id

    def get_node(self, node_id):
        self.nodes[node_id][self.NODE_LAST_VISIT] = self.cr_number
        return self.nodes[node_id]

    def get_leaf_covering_label(self, leaf_id):
        """ method returns covering label of
        a node with id leaf_id
        if the label is an integer then we return list of lenght N
        filled with the integer
        """
        labels = self.nodes[leaf_id][self.NODE_COVERING_LABEL]
        if not isinstance(labels, list):
            labels = [labels] * self.N
        return labels

    def set_covering_label(self, leaf_id, new_label):
        """ method sets covering label for a leaf with id leaf_id,
        new_label can be an integer of a list
        """
        node = []
        # adding last visit
        node.append(self.cr_number)
        # adding covering label
        node.append(new_label)
        self.nodes[leaf_id] = node

    def add_edge(self, src_node_id, dest_node_id, tokens):
        if not self.edges.has_key(src_node_id):
            self.edges[src_node_id] = {}
        new_edge = []
        new_edge.append(src_node_id)
        new_edge.append(dest_node_id)
        new_edge.append(tokens[:])
        self.edges[src_node_id][tokens[0]] = new_edge

    def get_edge(self, src_node_id, first_token):
        e = self.edges[src_node_id][first_token]
        # remember last visit
        self.get_node(src_node_id)
        self.get_node(e[self.EDGE_DEST_NODE_ID])
        return e

    def edge_exist(self, src_node_id, first_token):
        """ method returns true if the edge exists in the tree
        """
        return (self.edges.has_key(src_node_id) and
                    self.edges[src_node_id].has_key(first_token))

    def split_edge(self, src_node_id, first_token, split_point):
        """ src_node_id and first_token determines the edge to split
        split_point is lenth of tokens in firs part of the edge
        method returns id of a new node that splits old edge
        """
        e = self.get_edge(src_node_id, first_token)
        # shrinking the edge
        e_dest_node_id_old = e[self.EDGE_DEST_NODE_ID]
        middle_node_id = self.add_node()
        first_part_of_e = e[self.EDGE_TOKENS][:split_point]
        second_part_of_e = e[self.EDGE_TOKENS][split_point:]
        self.add_edge(src_node_id, middle_node_id, first_part_of_e)
        # creating 2nd part of the edge
        self.add_edge(middle_node_id, e_dest_node_id_old, second_part_of_e)
        return middle_node_id

    def calculate_covering_and_add_branches(self):
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
            leaf_id = self.get_leaf_nodeid_or_add_it(s)
            list_of_leaves_id.append(leaf_id)
            matrix_labels[i] = self.get_leaf_covering_label(leaf_id)
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

    def get_leaf_nodeid_or_add_it(self, s):
        """ given string s (list of tokens) of length N
        the method returns leaf id that string path
        the root -> the leaf equals s
        If there is no such leaf, then
        method add it to the tree and sets
        covering label of the leaf as current revision number
        """
        # we need tom "match" substring s along the tree
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
            if not self.edge_exist(current_node_id, rev_tokens[0]):
                new_node_id = self.add_node()
                self.add_edge(current_node_id, new_node_id, rev_tokens)
                # adding covering label to the new created leaf
                self.set_covering_label(new_node_id, self.cr_number)
                return new_node_id
            # we can traverse down
            e = self.get_edge(current_node_id, rev_tokens[0])
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
            middle_node_id = self.split_edge(current_node_id,
                                             rev_tokens[0],
                                             prefix_len)
            # creating new edge with rev_tokens
            new_node_id = self.add_node()
            self.add_edge(middle_node_id,
                          new_node_id,
                          rev_tokens[prefix_len:])
            # adding covering label to the new created leaf
            self.set_covering_label(new_node_id, self.cr_number)
            return new_node_id

    def update_covering_labels(self, list_of_leaves_id):
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
            self.set_covering_label(leaf_id, labels)

    def update_revision_id_table(self):
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
            tuple2add = (self.rev_incremental_number, self.cr_number)
            self.revision_id_table.append(tuple2add)
            return
        revdist = self.rev_incremental_number - self.revision_id_table[-1][0]
        if revdist >= minrev:
            tuple2add = (self.rev_incremental_number, self.cr_number)
            self.revision_id_table.append(tuple2add)
        # deleting rule: delete tuple from the table
        # if distance between current revision and tuple is more than
        # max(self.K_rev, number_of_revisions_in_timetable) + minrev revisions
        if len(self.timetable) == 0:
            th = 0
        else:
            inc_number_oldest = self.timetable[0][0]
            th = self.rev_incremental_number - inc_number_oldest
        th = max(self.K_revisions, th) + minrev
        while True:
            revdist = self.rev_incremental_number - self.revision_id_table[0][0]
            if revdist > th:
                del self.revision_id_table[0]
            else:
                break

    def update_timetable(self, timestamp=None):
        """ time table is needed for
        truncating nodes of the tree.
        timetable is a list of tuples
        (revision index number, timestamp)
        add new elemnts to the end
        [oldest ----- most recent]
        """
        self.cr_timestamp = timestamp
        if timestamp == None:
            return
        # adding rule: if day distance between
        # current revsion and previous inserted revision
        # is more or equal than mindays days then add new tuple
        # to the time list
        mindays = 5
        if len(self.timetable) == 0:
            tuple2add = (self.rev_incremental_number, self.cr_timestamp)
            self.timetable.append(tuple2add)
            return
        if self.time_distance(self.timetable[-1][1],
                              self.cr_timestamp) >= mindays:
            tuple2add = (self.rev_incremental_number, self.cr_timestamp)
            self.timetable.append(tuple2add)
        # deleting rule: delete tuple from the timetable
        # if distance between current revision and tuple is
        # more thatn self.K_time+mindays days
        while True:
            if self.time_distance(self.timetable[0][1],
                                  self.cr_timestamp) > self.K_time + mindays:
                del self.timetable[0]
            else:
                break

    def get_threshold(self):
        """ method calculates threshold for truncating the tree.

        note that revision_id_table stores maximum between K_rev revisions and
        number of revisions stored in the timetable, so threshold is the oldest
        revision id stored in revision_id_table
        """
        return self.revision_id_table[0][1]

    def truncate(self, threshold):
        """ the function delete nodes and edges.
        all nodes have revision id when they were visited last time.
        nodes and edges which were visited last time
        before threshold will be deleted.
        """
        # todo(michael): rewrite function so it is more readable, also
        #                make sure that it works arbitrary revision numbers,
        #                not just incremental.
        # sorted list of pairs - id and how many
        # revisions ago it was visited
        # sorted by second parameter - revisions between last visit
        threshold = self.cr_number - threshold
        #print 'real threshold is ', threshold
        l = [(x, self.cr_number - y[self.NODE_LAST_VISIT]) for x, y
                              in sorted(self.nodes.iteritems(),
                                        key = lambda (x,y): self.cr_number
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

    def get_new_node_id(self):
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

    def time_distance(self, x1, x2):
        """ x1 and x2 are timestamps in 'yyyymmddhhmmss' format
        returns |x2 - x1| in days.
        """
        year1 = int(x1[:4])
        month1 = int(x1[4:6])
        day1 = int(x1[6:8])
        year2 = int(x2[:4])
        month2 = int(x2[4:6])
        day2 = int(x2[6:8])
        dist = (365 * (year2 - year1) +
                30 * (month2 - month1) +
                day2 - day1)
        return abs(dist)

    def is_leaf(self, node_id):
        return node_id not in self.edges

    def dump2json(self):
        """ method returns json string which stores the tree and
        extra info (last_inserted_revision_id)
        format of the string is next:
        "[last_inserted_revision_id, json_representing_the_tree]"
        """
        tree_dict = self.get_tree_as_dict()
        res = [self.last_inserted_revision_id, tree_dict]
        return cjson.encode(res)

    def get_tree_as_dict(self):
        """ method returns dictionary which represents the tree and
        contains some class values
        json representation of the tree dictionary is next
        nodeid -> [last visit, [tokens, XXX] ]
        where XXX is destination node id if edge
        with tokens leads to non leaf node
        or XXX is [last visit, labels of tokens]
        where labels of tokens is tuple (l_1, ..., l_N)
        labels of tokens on the path root->the leaf,
        if l_1=...=l_N=l then instead of tuple store only number l
        """
        # convering x to str(x) because
        # fast module cjson for serialization
        # accepts only keys of type string
        res = {}
        for x in self.nodes:
            if not self.is_leaf(x):
                l = [self.nodes[x][self.NODE_LAST_VISIT]]
                res[str(x)] = l
        for x in self.edges:
            for y in self.edges[x]:
                e = self.edges[x][y]
                res[str(x)].append(' '.join(e[self.EDGE_TOKENS]))
                # Test if edge leads to leaf.
                dest = e[self.EDGE_DEST_NODE_ID]
                if self.is_leaf(dest):
                    n_dest = self.nodes[dest]
                    res[str(x)].append(n_dest[self.NODE_LAST_VISIT])
                    res[str(x)].append(n_dest[self.NODE_COVERING_LABEL])
                else:
                    res[str(x)].append(dest)
        # adding another values like N, K_rev, K_time, current timestamp
        class_values = {}
        class_values['N'] = self.N
        class_values['K_revisions'] = self.K_revisions
        class_values['K_time'] = self.K_time
        class_values['cr_timestamp'] = self.cr_timestamp
        class_values['lin_id'] = self.lin_id
        class_values['rev_incremental_number'] = self.rev_incremental_number
        res['class_values'] = class_values
        # addin timetable and revision_id_table
        res['timetable'] = self.timetable
        res['revision_id_table'] = self.revision_id_table
        return res

    def get_tree_json(self, json_string):
        """method returns json which represents the tree
        """
        idx = json_string.find('{')
        return json_string[idx : -1]

    @staticmethod
    def get_last_revision_id(json_string):
        """ method returns last inserted revision id given json
        which  stores information about the ojbect TriesA5
        """
        idx = json_string.find(',')
        return int(json_string[1 : idx])

    def load_from_json(self, json_string):
        """ method loads the tree from json string
        About id of leaf nodes:
        we are losing ids of leaf nodes while dumping to json
        because leaf node cannot become internal node during
        tree construction, then we are assigning negative integers
        as id to leaf nodes (to prevent fast increasing self.lin_id)
        """
        tree_json = self.get_tree_json(json_string)
        nodes_edges = cjson.decode(tree_json)
        #nodes_edges = json.loads(tree_json)
        # first we extract class_values
        class_values = nodes_edges['class_values']
        del(nodes_edges['class_values'])
        self.N = class_values['N']
        self.K_revisions = class_values['K_revisions']
        self.K_time = class_values['K_time']
        self.cr_timestamp = class_values['cr_timestamp']
        self.lin_id = class_values['lin_id']
        self.rev_incremental_number = class_values['rev_incremental_number']
        # loading timetable and revision_id_table
        self.timetable = nodes_edges['timetable']
        self.revision_id_table = nodes_edges['revision_id_table']
        del(nodes_edges['timetable'])
        del(nodes_edges['revision_id_table'])
        # last inserted node id
        #lin_id = 0
        # last inserted leaf id
        lil_id = 0
        nodes = {}
        edges = {}
        for x in nodes_edges:
            src_node_id = int(x)
            #lin_id = max(lin_id, src_node_id)
            line = nodes_edges[x]
            nodes[src_node_id] = [line[0]]
            edges[src_node_id] = {}
            idx = 1
            while idx < len(line):
                # in the beginning of each loop idx point
                # to a string of tokens, so line[idx] is
                # a string with tokens
                if not isinstance(line[idx], basestring):
                    raise Exception('error in parsing json')
                tokens = line[idx].split()
                # edge leads to a leaf
                if ((idx + 2) < len(line) and
                   not isinstance(line[idx + 2], basestring)):
                    # updating leaf id
                    lil_id -= 1
                    # adding a node
                    nodes[lil_id] = [line[idx + 1], line[idx + 2]]
                    # adding an edge
                    e = [src_node_id, lil_id, tokens]
                    edges[src_node_id][tokens[0]] = e
                    idx += 3
                # edge leads to an internal node
                else:
                    # creating an edge
                    e = [src_node_id, line[idx + 1], tokens]
                    edges[src_node_id][tokens[0]] = e
                    idx += 2
        # at this point we have nodes and edges
        self.nodes = nodes
        self.edges = edges
        self.last_inserted_revision_id = self.get_last_revision_id(json_string)
        #self.lin_id = lin_id

    def dump_covering2json(self):
        """ precondition is self.cr_covering is
        a covering of current revision
        """
        return cjson.encode(self.cr_covering[self.N : -self.N])


def plot_TriesA5(nodes, edges, revisions, file_name):
    ''' file_name is a name of a file to save image
    nonleaf nodes have text: node_id(node last visit)
    leaf nodes have text: [covering label](node last visit)
    '''
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
