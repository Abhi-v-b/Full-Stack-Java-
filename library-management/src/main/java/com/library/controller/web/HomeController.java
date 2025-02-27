package com.library.controller.web;

import com.library.service.BookService;
import com.library.service.UserService;
import com.library.service.BorrowingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    private final BookService bookService;
    private final UserService userService;
    private final BorrowingService borrowingService;

    @Autowired
    public HomeController(BookService bookService, 
                         UserService userService, 
                         BorrowingService borrowingService) {
        this.bookService = bookService;
        this.userService = userService;
        this.borrowingService = borrowingService;
    }

    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("totalBooks", bookService.getAllBooks().size());
        model.addAttribute("activeUsers", userService.getActiveUsers().size());
        model.addAttribute("activeBorrowings", borrowingService.getCurrentBorrowings().size());
        model.addAttribute("recentBorrowings", borrowingService.getCurrentBorrowings());
        model.addAttribute("overdueBorrowings", borrowingService.getOverdueBorrowings());
        return "index";
    }
} 