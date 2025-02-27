package com.library.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.library.entity.Borrowing;
import com.library.entity.User;
import com.library.entity.Book;
import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface BorrowingRepository extends JpaRepository<Borrowing, Long> {
    List<Borrowing> findByUser(User user);
    List<Borrowing> findByBook(Book book);
    List<Borrowing> findByStatus(String status);
    List<Borrowing> findByDueDateBeforeAndStatus(LocalDateTime date, String status);
    List<Borrowing> findByUserAndStatus(User user, String status);
} 