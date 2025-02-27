package com.library.service.impl;

import com.library.entity.Book;
import com.library.entity.Borrowing;
import com.library.entity.User;
import com.library.repository.BorrowingRepository;
import com.library.repository.BookRepository;
import com.library.repository.UserRepository;
import com.library.service.BorrowingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class BorrowingServiceImpl implements BorrowingService {

    private final BorrowingRepository borrowingRepository;
    private final BookRepository bookRepository;
    private final UserRepository userRepository;

    @Autowired
    public BorrowingServiceImpl(BorrowingRepository borrowingRepository,
                              BookRepository bookRepository,
                              UserRepository userRepository) {
        this.borrowingRepository = borrowingRepository;
        this.bookRepository = bookRepository;
        this.userRepository = userRepository;
    }

    @Override
    public Borrowing borrowBook(Long userId, Long bookId, Integer durationInDays) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new RuntimeException("Book not found with id: " + bookId));

        if (book.getAvailableQuantity() <= 0) {
            throw new RuntimeException("Book is not available for borrowing");
        }

        Borrowing borrowing = new Borrowing();
        borrowing.setUser(user);
        borrowing.setBook(book);
        borrowing.setBorrowDate(LocalDateTime.now());
        borrowing.setDueDate(LocalDateTime.now().plusDays(durationInDays));
        borrowing.setStatus("BORROWED");

        book.setAvailableQuantity(book.getAvailableQuantity() - 1);
        bookRepository.save(book);

        return borrowingRepository.save(borrowing);
    }

    @Override
    public Borrowing returnBook(Long borrowingId) {
        Borrowing borrowing = borrowingRepository.findById(borrowingId)
                .orElseThrow(() -> new RuntimeException("Borrowing not found with id: " + borrowingId));

        if ("RETURNED".equals(borrowing.getStatus())) {
            throw new RuntimeException("Book is already returned");
        }

        Book book = borrowing.getBook();
        book.setAvailableQuantity(book.getAvailableQuantity() + 1);
        bookRepository.save(book);

        borrowing.setReturnDate(LocalDateTime.now());
        borrowing.setStatus("RETURNED");
        return borrowingRepository.save(borrowing);
    }

    @Override
    public Optional<Borrowing> getBorrowingById(Long id) {
        return borrowingRepository.findById(id);
    }

    @Override
    public List<Borrowing> getAllBorrowings() {
        return borrowingRepository.findAll();
    }

    @Override
    public List<Borrowing> getBorrowingsByUser(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));
        return borrowingRepository.findByUser(user);
    }

    @Override
    public List<Borrowing> getBorrowingsByBook(Long bookId) {
        Book book = bookRepository.findById(bookId)
                .orElseThrow(() -> new RuntimeException("Book not found with id: " + bookId));
        return borrowingRepository.findByBook(book);
    }

    @Override
    public List<Borrowing> getOverdueBorrowings() {
        return borrowingRepository.findByDueDateBeforeAndStatus(LocalDateTime.now(), "BORROWED");
    }

    @Override
    public List<Borrowing> getCurrentBorrowings() {
        return borrowingRepository.findByStatus("BORROWED");
    }

    @Override
    public boolean isBookAvailableForBorrowing(Long bookId) {
        return bookRepository.findById(bookId)
                .map(book -> book.getAvailableQuantity() > 0)
                .orElse(false);
    }

    @Override
    public boolean hasUserExceededBorrowingLimit(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));
        
        // Get current borrowings for the user
        List<Borrowing> currentBorrowings = borrowingRepository.findByUserAndStatus(user, "BORROWED");
        
        // You can adjust the borrowing limit as needed
        int borrowingLimit = 5;
        return currentBorrowings.size() >= borrowingLimit;
    }

    @Override
    public void notifyOverdueBooks() {
        // Implementation for sending notifications about overdue books
        // This could involve sending emails, SMS, or other notifications
        List<Borrowing> overdueBorrowings = getOverdueBorrowings();
        for (Borrowing borrowing : overdueBorrowings) {
            // Add notification logic here
            System.out.println("Overdue notification for borrowing ID: " + borrowing.getId());
        }
    }
} 