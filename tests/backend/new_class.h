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
  class View;
  struct Window;
  struct Shape;
  enum class Access { rw, ro, wo };
  enum class Type { Integer, FloatingPoint, String, Bool };

  template <typename VariantType>
  Register(std::string const& name, //
           Access access,             //
           std::vector<std::vector<VariantType> > data);

  Register(std::string const& name, //
           Access mode,             //
           Type type,               //
           Shape shape);

  ~Register();

  template <typename UserType> //
  std::vector<std::vector<UserType> > read();

  template <typename UserType> //
  void write(std::vector<std::vector<UserType> > data);

  std::string getName();

  Shape getShape();

  Access getAccessMode();

  Type getType();

  View getView(Window w);

  struct Shape {
    size_t rows;
    size_t columns;
  };
  struct Window {
    Shape shape;
    size_t row_offset;
    size_t column_offset;
  };

  class View {
  public:
    View(Register& e, Window w);
    View(Register& e);

    template <typename UserType> //
    std::vector<std::vector<UserType> > read();
    template <typename UserType>
    void write(std::vector<std::vector<UserType> >& d);
  };
};

//template std::vector<std::vector<int> > Register::read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();
//template std::vector<std::vector<int> > read();




} // namespace TestBackend

