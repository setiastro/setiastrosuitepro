import ast
import builtins
import sys
import os

def check_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filename)
        except SyntaxError as e:
            print(f"SyntaxError in {filename}: {e}")
            return

    defined = set(dir(builtins))
    defined.add("__file__")
    defined.add("__name__")
    defined.add("__doc__")
    defined.add("__package__")
    
    # Add common PyQt/Application globals if needed, or loosely check
    # But for "precise" check we want to catch real errors.
    
    undefined = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_scope = defined.copy()
            self.scopes = [self.current_scope]

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                # if 'import foo.bar as baz', baz is defined
                # if 'import foo', foo is defined
                self.current_scope.add(name.split('.')[0])

        def visit_ImportFrom(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                if name == '*':
                    # Wildcard import - tricky. 
                    # We can't know what's imported without analyzing the module.
                    # For now, we might have to ignore this or mark it.
                    pass 
                else:
                    self.current_scope.add(name)

        def visit_FunctionDef(self, node):
            self.current_scope.add(node.name)
            # Enter scope
            parent_scope = self.current_scope
            self.current_scope = self.current_scope.copy()
            self.scopes.append(self.current_scope)
            
            # Arguments are defined
            for arg in node.args.args:
                self.current_scope.add(arg.arg)
            if node.args.vararg: self.current_scope.add(node.args.vararg.arg)
            if node.args.kwarg: self.current_scope.add(node.args.kwarg.arg)
            
            self.generic_visit(node)
            
            # Exit scope
            self.scopes.pop()
            self.current_scope = parent_scope

        def visit_ClassDef(self, node):
            self.current_scope.add(node.name)
            # Enter scope
            parent_scope = self.current_scope
            self.current_scope = self.current_scope.copy()
            self.scopes.append(self.current_scope)
            
            self.generic_visit(node)
            
            # Exit scope
            self.scopes.pop()
            self.current_scope = parent_scope

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                self.current_scope.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                if node.id not in self.current_scope:
                    # Look in parent scopes
                    found = False
                    for scope in reversed(self.scopes):
                        if node.id in scope:
                            found = True
                            break
                    if not found:
                        undefined.append((node.lineno, node.id))

        def visit_Attribute(self, node):
            self.visit(node.value)
            # We don't check the attribute itself (node.attr) because it depends on the object

    Visitor().visit(tree)
    
    if undefined:
        print(f"Undefined in {filename}:")
        for lineno, name in sorted(list(set(undefined))): # dedup
            print(f"  Line {lineno}: {name}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".py"):
                        check_file(os.path.join(root, file))
        else:
            check_file(path)
