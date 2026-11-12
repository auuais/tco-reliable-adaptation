"""Add one optional fine-alignment objective callback to the pinned upstream optimizer."""
import ast
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

EXPECTED_SOURCE='d811614a10fe7920fb41285c3b8b9162b38a5e1ea68823f76383a2f209d3a49f'


def compile_hook(function):
    source=inspect.getsource(function)
    tree=ast.parse(source)
    definition=tree.body[0]
    if not isinstance(definition,ast.FunctionDef) or definition.name!='sparse_scene_optimizer':
        raise ValueError('Unexpected optimizer function')
    if 'prior_callback' in inspect.signature(function).parameters:
        raise ValueError('Optimizer already has the research hook')
    definition.args.kwonlyargs.append(ast.arg(arg='prior_callback'))
    definition.args.kw_defaults.append(ast.Constant(None))
    assignments=[node for node in ast.walk(definition)
                 if isinstance(node,ast.Assign) and len(node.targets)==1
                 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='loss'
                 and any(isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id=='loss_func'
                         for child in ast.walk(node.value))]
    if len(assignments)!=1:
        raise ValueError('Expected exactly one upstream optimization objective')
    assignment=assignments[0]
    original=assignment.value
    callback=ast.Call(func=ast.Name(id='prior_callback',ctx=ast.Load()),
                      args=[ast.Name(id=name,ctx=ast.Load()) for name in ('K','cam2w','depthmaps')],
                      keywords=[ast.keyword(arg='phase',value=ast.Attribute(value=ast.Name(id='loss_func',ctx=ast.Load()),attr='__name__',ctx=ast.Load())),
                                ast.keyword(arg='iteration',value=ast.Name(id='iter',ctx=ast.Load()))])
    assignment.value=ast.IfExp(
        test=ast.Compare(left=ast.Name(id='prior_callback',ctx=ast.Load()),ops=[ast.Is()],comparators=[ast.Constant(None)]),
        body=original,orelse=ast.BinOp(left=original,op=ast.Add(),right=callback))
    ast.fix_missing_locations(tree)
    namespace=dict(function.__globals__)
    exec(compile(tree,inspect.getsourcefile(function),'exec'),namespace)
    return namespace[definition.name]


@contextmanager
def optimizer_hook(module,callback_factory=None):
    original=module.sparse_scene_optimizer
    path=Path(inspect.getsourcefile(original))
    digest=hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
    if digest!=EXPECTED_SOURCE:
        raise ValueError('Upstream sparse alignment source changed')
    compiled=compile_hook(original)
    callbacks=[]
    signature=inspect.signature(original)

    def optimizer(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        bound.apply_defaults()
        callback=None if callback_factory is None else callback_factory(bound.arguments)
        if callback is not None:
            callbacks.append(callback)
        return compiled(*args,prior_callback=callback,**kwargs)

    module.sparse_scene_optimizer=optimizer
    try:
        yield callbacks
    finally:
        module.sparse_scene_optimizer=original
