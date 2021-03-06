#pragma once

#include <memory>
#include <string>
#include <vector>

#include "variant_types.h"

namespace TestBackend {

  class Register {
   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

   public:
    class View;
    class Shape;
    struct Window;
    enum class Access { rw, ro, wo };
    enum class Type { Integer = 0, FloatingPoint, Bool, String };

    template<typename VariantType>
    Register(std::string const& name, //
        Access access,                //
        std::vector<std::vector<VariantType>>
            data);
    Register(std::string const& name, //
        Access mode,                  //
        Type type,                    //
        Shape shape = {1, 1});
    Register(Register const& r);
    Register(Register&& r);
    ~Register();

    std::string getName() const;
    Shape getShape() const;
    Access getAccessMode() const;
    Type getType() const;
    View getView(Window w);

    template<typename UserType> //
    std::vector<std::vector<UserType>> read();
    template<typename UserType> //
    void write(std::vector<std::vector<UserType>> const& data);

    class Shape {
     private:
      size_t rows_;
      size_t columns_;

     public:
      Shape(size_t rows, size_t columns);
      bool operator==(Shape const& rhs);
      bool operator!=(Shape const& rhs);

      size_t rowSize() const { return rows_; }
      size_t columnSize() const { return columns_; }
    };
    struct Window {
      Shape shape;
      size_t row_offset;
      size_t column_offset;
    };
    class View {
     private:
      struct Impl;
      std::unique_ptr<Impl> impl_;

     public:
      View(Register& r, Window w);
      View(View const& v);
      View(View&& v);
      ~View();

      template<typename UserType> //
      std::vector<std::vector<UserType>> read();
      template<typename UserType>
      void write(std::vector<std::vector<UserType>> const& d);

      friend std::string registerName(Register::View const& v);
      friend Register::Access getAccessMode(Register::View const& v);
      friend size_t columns(Register::View const& v);
      friend size_t rows(Register::View const& v);
    };
  };
  size_t columns(Register const& r);
  size_t rows(Register const& r);
} // namespace TestBackend
