package com.library.service.impl;

import com.library.entity.Book;
import com.library.repository.BookRepository;
import com.library.service.BookService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class BookServiceImpl implements BookService {
    private final BookRepository bookRepository;

    @Autowired
    public BookServiceImpl(BookRepository bookRepository) {
        this.bookRepository = bookRepository;
    }

    @Override
    public Book saveBook(Book book) {
        // Log the save operation
        System.out.println("Saving book: " + book.getTitle());
        
        // Set initial available quantity if not set
        if (book.getAvailableQuantity() == null) {
            book.setAvailableQuantity(book.getQuantity());
        }
        
        try {
            Book savedBook = bookRepository.save(book);
            System.out.println("Book saved successfully with ID: " + savedBook.getId());
            return savedBook;
        } catch (Exception e) {
            System.err.println("Error saving book: " + e.getMessage());
            throw e;
        }
    }

    @Override
    public Optional<Book> getBookById(Long id) {
        return bookRepository.findById(id);
    }

    @Override
    public Optional<Book> getBookByIsbn(String isbn) {
        return bookRepository.findByIsbn(isbn);
    }

    @Override
    public List<Book> getAllBooks() {
        return bookRepository.findAll();
    }

    @Override
    public Book updateBook(Long id, Book bookDetails) {
        return bookRepository.findById(id)
            .map(existingBook -> {
                existingBook.setTitle(bookDetails.getTitle());
                existingBook.setAuthor(bookDetails.getAuthor());
                existingBook.setIsbn(bookDetails.getIsbn());
                existingBook.setPublicationYear(bookDetails.getPublicationYear());
                existingBook.setPublisher(bookDetails.getPublisher());
                existingBook.setCategory(bookDetails.getCategory());
                existingBook.setDescription(bookDetails.getDescription());
                
                Integer oldQuantity = existingBook.getQuantity();
                Integer newQuantity = bookDetails.getQuantity();
                Integer oldAvailable = existingBook.getAvailableQuantity();
                Integer borrowedBooks = oldQuantity - oldAvailable;
                
                existingBook.setQuantity(newQuantity);
                existingBook.setAvailableQuantity(newQuantity - borrowedBooks);
                existingBook.setLocation(bookDetails.getLocation());
                
                return bookRepository.save(existingBook);
            })
            .orElseThrow(() -> new RuntimeException("Book not found with id: " + id));
    }

    @Override
    public void deleteBook(Long id) {
        bookRepository.deleteById(id);
    }

    @Override
    public boolean existsByIsbn(String isbn) {
        return bookRepository.findByIsbn(isbn).isPresent();
    }

    @Override
    public List<Book> searchBooks(String keyword) {
        if (keyword == null || keyword.trim().isEmpty()) {
            return getAllBooks();
        }
        
        List<Book> results = new ArrayList<>();
        results.addAll(bookRepository.findByTitleContainingIgnoreCase(keyword));
        results.addAll(bookRepository.findByAuthorContainingIgnoreCase(keyword));
        return results.stream().distinct().toList();
    }

    @Override
    public List<Book> getBooksByCategory(String category) {
        return bookRepository.findByCategory(category);
    }

    @Override
    public boolean isBookAvailable(Long id) {
        return bookRepository.findById(id)
            .map(book -> book.getAvailableQuantity() > 0)
            .orElse(false);
    }
} 