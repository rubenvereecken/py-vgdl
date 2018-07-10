from .tools import Node, indentTreeParser
from .core import BasicGame
from vgdl import registry, ontology
from vgdl.core import BasicGame

registry.register_all(ontology)
registry.register_class(BasicGame)


class VGDLParser:
    """ Parses a string into a Game object. """
    verbose = False

    def parseGame(self, tree, **kwargs):
        """ Accepts either a string, or a tree. """
        if not isinstance(tree, Node):
            tree = indentTreeParser(tree).children[0]
        sclass, args = self._parseArgs(tree.content)
        args.update(kwargs)
        self.game = sclass(**args)
        for c in tree.children:
            if c.content == "SpriteSet":
                self.parseSprites(c.children)
            if c.content == "InteractionSet":
                self.parseInteractions(c.children)
            if c.content == "LevelMapping":
                self.parseMappings(c.children)
            if c.content == "TerminationSet":
                self.parseTerminations(c.children)

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

    def parseInteractions(self, inodes):
        for inode in inodes:
            if ">" in inode.content:
                pair, edef = [x.strip() for x in inode.content.split(">")]
                eclass, args = self._parseArgs(edef)
                objs = [x.strip() for x in pair.split(" ") if len(x)>0]
                for obj in objs[1:]:
                    self.game.collision_eff.append(tuple([objs[0], obj, eclass, args]))
                if self.verbose:
                    print("Collision", pair, "has effect:", edef)

    def parseTerminations(self, tnodes):
        for tn in tnodes:
            sclass, args = self._parseArgs(tn.content)
            if self.verbose:
                print("Adding:", sclass, args)
            self.game.terminations.append(sclass(**args))

    def parseSprites(self, snodes, parentclass=None, parentargs={}, parenttypes=[]):
        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parseArgs(sdef, parentclass, parentargs.copy())
            stypes = parenttypes+[key]
            if 'singleton' in args:
                if args['singleton']==True:
                    self.game.singletons.append(key)
                args = args.copy()
                del args['singleton']

            if len(sn.children) == 0:
                if self.verbose:
                    print("Defining:", key, sclass, args, stypes)
                self.game.sprite_constr[key] = (sclass, args, stypes)
                if key in self.game.sprite_order:
                    # last one counts
                    self.game.sprite_order.remove(key)
                self.game.sprite_order.append(key)
            else:
                self.parseSprites(sn.children, sclass, args, stypes)

    def parseMappings(self, mnodes):
        for mn in mnodes:
            c, val = [x.strip() for x in mn.content.split(">")]
            assert len(c) == 1, "Only single character mappings allowed."
            # a char can map to multiple sprites
            keys = [x.strip() for x in val.split(" ") if len(x)>0]
            if self.verbose:
                print("Mapping", c, keys)
            self.game.char_mapping[c] = keys

    def _parseArgs(self, s,  sclass=None, args=None):
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
