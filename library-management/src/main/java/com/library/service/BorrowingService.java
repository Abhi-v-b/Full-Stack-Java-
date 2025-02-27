package com.library.service;

import com.library.entity.Borrowing;
import com.library.entity.User;
import com.library.entity.Book;
import java.util.List;
import java.util.Optional;

public interface BorrowingService {
    Borrowing borrowBook(Long userId, Long bookId, Integer durationInDays);
    Borrowing returnBook(Long borrowingId);
    Optional<Borrowing> getBorrowingById(Long id);
    List<Borrowing> getAllBorrowings();
    List<Borrowing> getBorrowingsByUser(Long userId);
    List<Borrowing> getBorrowingsByBook(Long bookId);
    List<Borrowing> getOverdueBorrowings();
    List<Borrowing> getCurrentBorrowings();
    boolean isBookAvailableForBorrowing(Long bookId);
    boolean hasUserExceededBorrowingLimit(Long userId);
    void notifyOverdueBooks();
} 