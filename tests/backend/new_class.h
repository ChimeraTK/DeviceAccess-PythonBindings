#pragma once

#include <memory>
#include <string>
#include <vector>

#include "VariantTypes.h"

namespace TestBackend {

class Register {
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

public:
  struct Window;
  class View;
  class Shape;
  enum class Access { rw, ro, wo };
  enum class Type { Integer = 0, FloatingPoint, Bool, String };

  template <typename VariantType>
  Register(std::string const &name, //
           Access access,           //
           std::vector<std::vector<VariantType>> data);

  Register(std::string const &name, //
           Access mode,             //
           Type type,               //
           Shape shape = {1, 1});

  Register(Register &&r);
  ~Register();

  template <typename UserType> //
  std::vector<std::vector<UserType>> read();

  template <typename UserType> //
  void write(std::vector<std::vector<UserType>> const &data);

  std::string getName();
  Shape getShape();
  Access getAccessMode();
  Type getType();
  View getView(Window w);

  class Shape {
  private:
    size_t rows_;
    size_t columns_;

  public:
    Shape(size_t rows, size_t columns);
    size_t getRows(){return rows_;}
    size_t getColumns(){return columns_;}
    bool operator==(Shape const &rhs);
    bool operator!=(Shape const &rhs);
  };
  struct Window {
    Shape shape;
    size_t row_offset;
    size_t column_offset;
  };
  class View {
  public:
    View(Register &r, Window w);
    ~View();
    template <typename UserType> //
    std::vector<std::vector<UserType>> read();
    template <typename UserType>
    void write(std::vector<std::vector<UserType>> &d);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
  };
};

} // namespace TestBackend
