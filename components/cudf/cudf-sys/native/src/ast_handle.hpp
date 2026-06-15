#ifndef CUDF_RUST_AST_HANDLE_HPP
#define CUDF_RUST_AST_HANDLE_HPP

#include "ast.h"

#include <cudf/ast/expressions.hpp>

#include <memory>
#include <utility>
#include <vector>

struct cudf_ast_expression_owner {
  std::unique_ptr<cudf::ast::expression> expression;
  std::vector<std::shared_ptr<cudf_ast_expression_owner>> children;
};

using cudf_ast_expression_owner_ref = std::shared_ptr<cudf_ast_expression_owner>;

inline cudf_ast_expression_handle cudf_release_ast_expression_handle(
  std::unique_ptr<cudf::ast::expression> expression,
  std::vector<cudf_ast_expression_owner_ref> children = {})
{
  auto owner        = std::make_shared<cudf_ast_expression_owner>();
  owner->expression = std::move(expression);
  owner->children   = std::move(children);
  auto handle       = std::make_unique<cudf_ast_expression_owner_ref>(std::move(owner));
  return static_cast<cudf_ast_expression_handle>(handle.release());
}

inline cudf_ast_expression_owner_ref& cudf_ast_expression_owner_ref_from_handle(
  cudf_ast_expression_handle handle)
{
  return *static_cast<cudf_ast_expression_owner_ref*>(handle);
}

inline cudf::ast::expression const& cudf_ast_expression_from_handle(
  cudf_ast_expression_handle handle)
{
  return *cudf_ast_expression_owner_ref_from_handle(handle)->expression;
}

#endif  // CUDF_RUST_AST_HANDLE_HPP
