from types import FunctionType

from vgdl import registry, ontology
from vgdl.core import BasicGame, SpriteRegistry
from vgdl.core import Effect, FunctionalEffect

registry.register_all(ontology)
registry.register_class(BasicGame)


class Node(object):
    """ Lightweight indented tree structure, with automatic insertion at the right spot. """

    parent = None

    def __init__(self, content, indent, parent=None):
        self.children = []
        self.content = content
        self.indent = indent
        if parent:
            parent.insert(self)
        else:
            self.parent = None

    def insert(self, node):
        if self.indent < node.indent:
            if len(self.children) > 0:
                assert self.children[0].indent == node.indent, 'children indentations must match'
            self.children.append(node)
            node.parent = self
        else:
            assert self.parent, 'Root node too indented?'
            self.parent.insert(node)

    def __repr__(self):
        if len(self.children) == 0:
            return self.content
        else:
            return self.content + str(self.children)

    def get_root(self):
        if self.parent:
            return self.parent.get_root()
        else:
            return self


def indent_tree_parser(s, tabsize=8):
    """ Produce an unordered tree from an indented string. """
    # insensitive to tabs, parentheses, commas
    s = s.expandtabs(tabsize)
    s.replace('(', ' ')
    s.replace(')', ' ')
    s.replace(',', ' ')
    lines = s.split("\n")

    last = Node("", -1)
    for l in lines:
        # remove comments starting with "#"
        if '#' in l:
            l = l.split('#')[0]
        # handle whitespace and indentation
        content = l.strip()
        if len(content) > 0:
            indent = len(l) - len(l.lstrip())
            last = Node(content, indent, last)
    return last.get_root()


class VGDLParser:
    """ Parses a string into a Game object. """
    verbose = False

    def parse_game(self, tree, **kwargs):
        """ Accepts either a string, or a tree. """
        if not isinstance(tree, Node):
            tree = indent_tree_parser(tree).children[0]
        sclass, args = self._parse_args(tree.content)
        args.update(kwargs)
        # BasicGame construction
        self.sprite_registry = SpriteRegistry()
        self.game = sclass(self.sprite_registry, **args)
        for c in tree.children:
            # _, args = self._parseArgs(' '.join(c.content.split(' ')[1:]))
            if c.content.startswith("SpriteSet"):
                self.parse_sprites(c.children)
            if c.content == "InteractionSet":
                self.parse_interactions(c.children)
            if c.content == "LevelMapping":
                self.parse_mappings(c.children)
            if c.content == "TerminationSet":
                self.parse_terminations(c.children)

        return self.game

    def _eval(self, estr):
        """
        Whatever is visible in the global namespace (after importing the ontologies)
        can be used in the VGDL, and is evaluated.
        """
        # Classes and functions etc are registered with the global registry
        if estr in registry:
            return registry.request(estr)
        else:
            # Strings and numbers should just be interpreted
            return eval(estr)

    def parse_interactions(self, inodes):
        for inode in inodes:
            if ">" in inode.content:
                pair, edef = [x.strip() for x in inode.content.split(">")]
                eclass, kwargs = self._parse_args(edef)
                objs = [x.strip() for x in pair.split(" ") if len(x) > 0]

                # Create an effect for each actee
                for obj in objs[1:]:
                    args = [objs[0], obj]

                    if isinstance(eclass, FunctionType):
                        effect = FunctionalEffect(eclass, *args, **kwargs)
                    else:
                        assert issubclass(eclass, Effect)
                        effect = eclass(*args, **kwargs)

                    self.game.collision_eff.append(effect)

                if self.verbose:
                    print("Collision", pair, "has effect:", effect)

    def parse_terminations(self, tnodes):
        for tn in tnodes:
            sclass, args = self._parse_args(tn.content)
            if self.verbose:
                print("Adding:", sclass, args)
            self.game.terminations.append(sclass(**args))

    def parse_sprites(self, snodes, parentclass=None, parentargs={}, parenttypes=[]):
        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parse_args(sdef, parentclass, parentargs.copy())
            stypes = parenttypes + [key]

            if 'singleton' in args:
                if args['singleton'] == True:
                    self.game.sprite_registry.register_singleton(key)
                args = args.copy()
                del args['singleton']

            if len(sn.children) == 0:
                if self.verbose:
                    print("Defining:", key, sclass, args, stypes)
                self.sprite_registry.register_sprite_class(key, sclass, args, stypes)
                if key in self.game.sprite_order:
                    # last one counts
                    self.game.sprite_order.remove(key)
                self.game.sprite_order.append(key)
            else:
                self.parse_sprites(sn.children, sclass, args, stypes)

    def parse_mappings(self, mnodes):
        for mn in mnodes:
            c, val = [x.strip() for x in mn.content.split(">")]
            assert len(c) == 1, "Only single character mappings allowed."
            # a char can map to multiple sprites
            keys = [x.strip() for x in val.split(" ") if len(x) > 0]
            if self.verbose:
                print("Mapping", c, keys)
            self.game.char_mapping[c] = keys

    def _parse_args(self, s, sclass=None, args=None):
        if not args:
            args = {}
        sparts = [x.strip() for x in s.split(" ") if len(x) > 0]
        if len(sparts) == 0:
            return sclass, args
        if not '=' in sparts[0]:
            sclass = self._eval(sparts[0])
            sparts = sparts[1:]
        for sp in sparts:
            k, val = sp.split("=")
            try:
                args[k] = self._eval(val)
            except:
                args[k] = val
        return sclass, args
