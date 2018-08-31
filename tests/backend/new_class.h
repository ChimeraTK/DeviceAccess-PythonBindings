#pragma once

namespace TestBackend {
class DBaseElem_new {
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  class View;
  struct Window;
  struct Shape;
  enum class AccessMode;
  enum class RegisterType;

public:
  template <typename UserType>
  DBaseElem_new(std::string const& name, //
                AccessMode mode,         //
                RegisterType type,       //
                std::vector<std::vector<UserType> > data);
  DBaseElem_new(std::string const& name, //
                AccessMode mode,         //
                RegisterType type,       //
                Shape shape);

  template <typename UserType> //
  std::vector<std::vector<UserType> > read();
  template <typename UserType> //
  write(std::vector<std::vector<UserType> > d);

  std::string getName();
  Shape getShape();
  AccessMode getAccessMode();
  RegisterType getType();
  View getView(Window w);

  class View {
  public:
    View(DBaseElem_new& e, Window w);
    View(DBaseElem_new& e);

    template <typename UserType> //
    std::vector<std::vector<UsertType> > read();
    template <typename UserType>
    void write(std::vector<std::vector<UserType> >& d);
  };
  struct Window {
    Shape shape;
    size_t row_offset;
    size_t column_offset;
  };
  struct Shape {
    size_t rows;
    size_t columns;
    ;

    enum class AccessMode { rw, ro, wo };
    enum class RegisterType { Int, Double, String, Bool };
  };
} // TestBackend
