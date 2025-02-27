package com.library.service;

import com.library.entity.Book;
import java.util.List;
import java.util.Optional;

public interface BookService {
    Book saveBook(Book book);
    Optional<Book> getBookById(Long id);
    Optional<Book> getBookByIsbn(String isbn);
    List<Book> getAllBooks();
    Book updateBook(Long id, Book book);
    void deleteBook(Long id);
    boolean existsByIsbn(String isbn);
    List<Book> searchBooks(String keyword);
    List<Book> getBooksByCategory(String category);
    boolean isBookAvailable(Long id);
}